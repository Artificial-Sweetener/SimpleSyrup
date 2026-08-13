# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the U11 native-SDXL regional-LoRA visual acceptance matrix."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import (
    ManagedComfyServer,
    RunningManagedComfy,
)
from tools.sdxl_attention_coupling_integration.comfy_model_root import (
    resolve_active_comfy_model_root,
)
from tools.sdxl_attention_coupling_integration.managed_model_links import (
    ManagedModelLink,
    ManagedSdxlVisualModelLinks,
)
from tools.sdxl_attention_coupling_integration.visual_artifacts import (
    build_visual_review_sheets,
)
from tools.sdxl_attention_coupling_integration.visual_cases import (
    character_b_NAME,
    character_b_SOURCE,
    CHECKPOINT_NAME,
    ELDEN_STYLE_NAME,
    ELDEN_STYLE_SOURCE,
    character_c_NAME,
    character_c_SOURCE,
    checkpoint_a_SOURCE,
    SdxlVisualCase,
    visual_cases,
)
from tools.sdxl_attention_coupling_integration.visual_history import (
    decode_sdxl_visual_history,
)
from tools.sdxl_attention_coupling_integration.visual_launch import (
    sdxl_visual_sampling_launch_arguments,
)
from tools.sdxl_attention_coupling_integration.visual_masks import (
    ManagedSdxlVisualMasks,
)
from tools.sdxl_attention_coupling_integration.visual_results import (
    SdxlVisualResultRecorder,
)
from tools.sdxl_attention_coupling_integration.visual_workflow import (
    BuiltSdxlVisualWorkflow,
    build_sdxl_visual_workflow,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = Path(r"<COMFY_ROOT>")
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\universal-regional-adapter\u11"
)


def execute_visual_matrix(
    artifacts: IntegrationArtifacts,
    *,
    comfy_root: Path,
    readiness_timeout: float,
    prompt_timeout: float,
) -> Path:
    """Execute every case, preserve each result, and clean all exact owners."""

    recorder = SdxlVisualResultRecorder(artifacts.root)
    model_links = _model_links(comfy_root)
    masks = ManagedSdxlVisualMasks(
        input_root=comfy_root / "input",
        run_id=artifacts.run_id,
    )
    server_cleanup = False
    system_stats: JsonObject = {}
    mask_evidence: tuple[dict[str, object], ...] = ()
    with model_links:
        with masks:
            workflows = tuple(
                (
                    case,
                    build_sdxl_visual_workflow(
                        run_id=artifacts.run_id,
                        checkpoint_name=CHECKPOINT_NAME,
                        mask_names=masks.names(case.mask_profile),
                        case=case,
                    ),
                )
                for case in visual_cases()
            )
            mask_evidence = masks.evidence()
            required = frozenset(
                node_id
                for _case, workflow in workflows
                for node_id in workflow.required_node_ids
            )
            with ManagedComfyServer(
                comfy_root=comfy_root,
                artifacts=artifacts,
                required_node_ids=required,
                readiness_timeout=readiness_timeout,
                launch_arguments=sdxl_visual_sampling_launch_arguments(),
            ) as running:
                system_stats = running.system_stats
                for case, workflow in workflows:
                    LOGGER.info("Running U11 case %s: %s", case.case_id, case.label)
                    try:
                        _execute_case(
                            workflow,
                            case=case,
                            running=running,
                            recorder=recorder,
                            prompt_timeout=prompt_timeout,
                        )
                        build_visual_review_sheets(artifacts.root / "u11-result.json")
                    except BaseException as error:
                        recorder.record_failure(case, error)
                        raise
                port = running.port
                process = running.process
            server_cleanup = not process.is_running and is_loopback_port_available(port)
            artifacts.record_cleanup(
                process_running=process.is_running,
                port_available=is_loopback_port_available(port),
            )
    result = recorder.finalize(
        system_stats=system_stats,
        server_cleanup=server_cleanup,
        model_cleanup=model_links.cleaned,
        mask_cleanup=masks.cleaned,
        mask_evidence=mask_evidence,
    )
    build_visual_review_sheets(result)
    return result


def _execute_case(
    workflow: BuiltSdxlVisualWorkflow,
    *,
    case: SdxlVisualCase,
    running: RunningManagedComfy,
    recorder: SdxlVisualResultRecorder,
    prompt_timeout: float,
) -> None:
    """Submit, download, and durably record one exact case."""

    started_at = time.perf_counter()
    prompt_id = running.client.submit(workflow.prompt)
    history = running.client.wait_for_history(prompt_id, timeout=prompt_timeout)
    wall_runtime_ms = (time.perf_counter() - started_at) * 1000.0
    evidence = decode_sdxl_visual_history(history, workflow)
    images = {
        output.artifact_id: running.client.download_image(
            evidence[output.artifact_id].image_reference
        )
        for output in workflow.outputs
    }
    recorder.record_case(
        workflow,
        case=case,
        history=history,
        prompt_id=prompt_id,
        evidence=evidence,
        image_bytes=images,
        wall_runtime_ms=wall_runtime_ms,
    )


def _model_links(comfy_root: Path) -> ManagedSdxlVisualModelLinks:
    """Return the exact checkpoint and LoRA visibility owner."""

    return ManagedSdxlVisualModelLinks(
        model_root=resolve_active_comfy_model_root(comfy_root),
        links=(
            ManagedModelLink(checkpoint_a_SOURCE, "checkpoints", CHECKPOINT_NAME),
            ManagedModelLink(character_b_SOURCE, "loras", character_b_NAME),
            ManagedModelLink(character_c_SOURCE, "loras", character_c_NAME),
            ManagedModelLink(ELDEN_STYLE_SOURCE, "loras", ELDEN_STYLE_NAME),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command arguments and return one process exit status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--readiness-timeout", type=float, default=240.0)
    parser.add_argument("--prompt-timeout", type=float, default=1200.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = IntegrationArtifacts(args.output_root)
    try:
        result = execute_visual_matrix(
            artifacts,
            comfy_root=args.comfy_root,
            readiness_timeout=args.readiness_timeout,
            prompt_timeout=args.prompt_timeout,
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("Managed U11 SDXL visual matrix failed at %s", artifacts.root)
        return 1
    LOGGER.info("Managed U11 SDXL visual matrix completed: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
