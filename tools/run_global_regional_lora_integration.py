# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the P9.3 global/regional Anima LoRA matrix in managed ComfyUI."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tools.anima_attention_coupling_integration.workflow import (
    IntegrationWorkflowBuilder,
)
from tools.anima_attention_coupling_workflow import HEIGHT, WIDTH
from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.attention_coupling_benchmark.mask_artifacts import MaskArtifactWriter
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.default_paths import (
    default_benchmark_artifact_root,
    default_comfy_root,
)
from tools.comfy_integration.history_output import extract_saved_image
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.global_regional_lora_integration.matrix import MASK_CASE_ID, cases
from tools.global_regional_lora_integration.results import (
    GlobalRegionalLoraResultRecorder,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = default_comfy_root()
DEFAULT_OUTPUT_ROOT = default_benchmark_artifact_root(
    "anima-regional-prompting-v1/p9.3"
)


def execute_matrix(
    artifacts: IntegrationArtifacts,
    *,
    comfy_root: Path,
    readiness_timeout: float,
    prompt_timeout: float,
) -> Path:
    """Execute all success and intentional-rejection cases."""

    definitions = cases()
    manifest_case = next(
        case for case in load_manifest().cases if case.case_id == MASK_CASE_ID
    )
    mask_writer = MaskArtifactWriter(
        comfy_root / "input", f"p93-{artifacts.run_id.lower()}"
    )
    mask_names = mask_writer.write_case(manifest_case, width=WIDTH, height=HEIGHT)
    mask_paths = tuple((comfy_root / "input" / name).resolve() for name in mask_names)
    builder = IntegrationWorkflowBuilder()
    workflows = tuple(
        builder.build(
            case.integration,
            run_id=f"{artifacts.run_id}-{case.case_id}",
            mask_names=mask_names,
        )
        for case in definitions
    )
    required = frozenset().union(*(item.required_node_ids for item in workflows))
    recorder = GlobalRegionalLoraResultRecorder(artifacts.root)
    try:
        with ManagedComfyServer(
            comfy_root=comfy_root,
            artifacts=artifacts,
            required_node_ids=required,
            readiness_timeout=readiness_timeout,
        ) as running:
            for case, workflow in zip(definitions, workflows, strict=True):
                LOGGER.info(
                    "P9.3 starting %s — %s",
                    case.case_id,
                    case.integration.label,
                )
                prompt_id = running.client.submit(workflow.prompt)
                history = running.client.wait_for_history(
                    prompt_id,
                    timeout=prompt_timeout,
                )
                if case.expect_overlap_error:
                    recorder.record_overlap_rejection(
                        case,
                        workflow,
                        prompt_id=prompt_id,
                        history=history,
                    )
                    continue
                reference = extract_saved_image(history, workflow.save_node_id)
                image = running.client.download_image(reference)
                path = recorder.record_success(
                    case,
                    workflow,
                    prompt_id=prompt_id,
                    history=history,
                    reference=reference,
                    image_bytes=image,
                )
                LOGGER.info("P9.3 accepted %s", path)
            system_stats = running.system_stats
            port = running.port
            process = running.process
    finally:
        for mask_path in mask_paths:
            mask_path.unlink(missing_ok=True)
    masks_removed = not any(path.exists() for path in mask_paths)
    cleanup_verified = not process.is_running and is_loopback_port_available(port)
    artifacts.record_cleanup(
        process_running=process.is_running,
        port_available=is_loopback_port_available(port),
    )
    return recorder.finalize(
        definitions,
        system_stats=system_stats,
        cleanup_verified=cleanup_verified,
        masks_removed=masks_removed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and return one process status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--readiness-timeout", type=float, default=300.0)
    parser.add_argument("--prompt-timeout", type=float, default=1200.0)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    artifacts = IntegrationArtifacts(args.output_root)
    try:
        result = execute_matrix(
            artifacts,
            comfy_root=args.comfy_root,
            readiness_timeout=args.readiness_timeout,
            prompt_timeout=args.prompt_timeout,
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("P9.3 integration failed at %s", artifacts.root)
        return 1
    LOGGER.info("P9.3 integration completed at %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
