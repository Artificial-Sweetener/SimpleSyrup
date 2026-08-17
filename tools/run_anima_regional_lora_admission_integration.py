# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the P9.5 supported/rejected matrix in one managed ComfyUI."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from tools.anima_attention_coupling_integration.schema import (
    validate_public_node_metadata,
)
from tools.anima_regional_lora_admission_integration.graph_contract import (
    HEIGHT,
    MASK_CASE_ID,
    PUBLIC_NODE_ID,
    WIDTH,
)
from tools.anima_regional_lora_admission_integration.history import parse_history
from tools.anima_regional_lora_admission_integration.matrix import cases
from tools.anima_regional_lora_admission_integration.results import (
    RegionalLoraAdmissionResultRecorder,
)
from tools.anima_regional_lora_admission_integration.workflow import (
    RegionalLoraAdmissionWorkflowBuilder,
)
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

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = default_comfy_root()
DEFAULT_OUTPUT_ROOT = default_benchmark_artifact_root(
    "anima-regional-prompting-v1/p9.5"
)
FIXTURE_NODE_ID = "SimpleSyrupBenchmark.CreateRegionalHookFixture"


def execute_matrix(
    artifacts: IntegrationArtifacts,
    *,
    comfy_root: Path,
    readiness_timeout: float,
    prompt_timeout: float,
) -> Path:
    """Execute the complete ordered matrix and return its terminal result."""

    definitions = cases()
    manifest_case = next(
        case for case in load_manifest().cases if case.case_id == MASK_CASE_ID
    )
    mask_writer = MaskArtifactWriter(
        comfy_root / "input",
        f"p95-{artifacts.run_id.lower()}",
    )
    mask_names = mask_writer.write_case(manifest_case, width=WIDTH, height=HEIGHT)
    recorder = RegionalLoraAdmissionResultRecorder(artifacts.root)
    recorder.record_masks(mask_names, input_root=comfy_root / "input")
    builder = RegionalLoraAdmissionWorkflowBuilder(artifact_phase="p9.5")
    workflows = tuple(
        builder.build(case, run_id=artifacts.run_id, mask_names=mask_names)
        for case in definitions
    )
    required = frozenset().union(
        *(workflow.required_node_ids for workflow in workflows)
    )
    system_stats = {}
    port = 0
    process = None
    try:
        with ManagedComfyServer(
            comfy_root=comfy_root,
            artifacts=artifacts,
            required_node_ids=required,
            readiness_timeout=readiness_timeout,
        ) as running:
            public_metadata = running.client.node_metadata(PUBLIC_NODE_ID)
            validate_public_node_metadata(public_metadata)
            recorder.record_metadata(
                public_metadata,
                running.client.node_metadata(FIXTURE_NODE_ID),
            )
            for case, workflow in zip(definitions, workflows, strict=True):
                LOGGER.info("P9.5 starting %s — %s", case.case_id, case.label)
                started = time.perf_counter()
                prompt_id = running.client.submit(workflow.workflow.prompt)
                history = running.client.wait_for_history(
                    prompt_id,
                    timeout=prompt_timeout,
                )
                observed = parse_history(history, workflow)
                image_reference = None
                image_bytes = None
                if case.expect_success:
                    image_reference = extract_saved_image(
                        history,
                        workflow.workflow.save_node_id,
                    )
                    image_bytes = running.client.download_image(image_reference)
                image_path = recorder.record_case(
                    case,
                    workflow,
                    observed,
                    history=history,
                    prompt_id=prompt_id,
                    wall_runtime_ms=(time.perf_counter() - started) * 1000.0,
                    image_reference=image_reference,
                    image_bytes=image_bytes,
                )
                LOGGER.info(
                    "P9.5 accepted %s — %s",
                    case.label,
                    "pre-sampling rejection" if image_path is None else image_path,
                )
            system_stats = running.system_stats
            port = running.port
            process = running.process
    finally:
        _remove_owned_masks(comfy_root / "input", mask_names)
    if process is None:
        raise RuntimeError("P9.5 managed Comfy did not reach ready state.")
    process_stopped = not process.is_running
    port_available = is_loopback_port_available(port)
    masks_removed = all(
        not (comfy_root / "input" / name).exists() for name in mask_names
    )
    artifacts.record_cleanup(
        process_running=process.is_running,
        port_available=port_available,
    )
    return recorder.finalize(
        definitions,
        system_stats=system_stats,
        cleanup_verified=process_stopped and port_available and masks_removed,
    )


def _remove_owned_masks(input_root: Path, names: tuple[str, ...]) -> None:
    """Remove only exact uniquely prefixed masks created by this run."""

    resolved_root = input_root.resolve()
    for name in names:
        path = (resolved_root / name).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError("P9.5 mask cleanup escaped the Comfy input root.")
        path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse explicit boundaries and return one process exit status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--readiness-timeout", type=float, default=300.0)
    parser.add_argument("--prompt-timeout", type=float, default=1200.0)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
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
        LOGGER.exception("P9.5 integration failed at %s", artifacts.root)
        return 1
    LOGGER.info("P9.5 integration completed at %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
