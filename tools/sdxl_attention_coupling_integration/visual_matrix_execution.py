# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute an explicitly declared SDXL visual case sequence."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import (
    ManagedComfyServer,
    RunningManagedComfy,
)

from .comfy_model_root import resolve_active_comfy_model_root
from .managed_model_links import ManagedModelLink, ManagedSdxlVisualModelLinks
from .sampling_controls import SDXL_VISUAL_SAMPLING, validate_sdxl_visual_seed
from .visual_adapter_selections import (
    CHECKPOINT_SELECTION,
    LEFT_CHARACTER_SELECTION,
    RIGHT_CHARACTER_SELECTION,
    STYLE_SELECTION,
)
from .visual_artifacts import build_visual_review_sheets
from .visual_case_model import SdxlVisualCase
from .visual_history import decode_sdxl_visual_history
from .visual_inventory import SdxlVisualInventory
from .visual_launch import sdxl_visual_sampling_launch_arguments
from .visual_masks import ManagedSdxlVisualMasks
from .visual_results import SdxlVisualResultRecorder
from .visual_workflow import BuiltSdxlVisualWorkflow, build_sdxl_visual_workflow

LOGGER = logging.getLogger(__name__)


def execute_visual_cases(
    artifacts: IntegrationArtifacts,
    *,
    inventory: SdxlVisualInventory,
    cases: tuple[SdxlVisualCase, ...],
    comfy_root: Path,
    readiness_timeout: float,
    prompt_timeout: float,
    seed: int = SDXL_VISUAL_SAMPLING.seed,
) -> Path:
    """Execute every explicit case in one managed server trajectory."""

    if not cases:
        raise ValueError("SDXL visual execution requires at least one case.")
    validated_seed = validate_sdxl_visual_seed(seed)
    recorder = SdxlVisualResultRecorder(artifacts.root, cases=cases)
    model_links = build_sdxl_visual_model_links(comfy_root, inventory)
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
                        checkpoint_name=CHECKPOINT_SELECTION,
                        mask_names=masks.names(case.mask_profile),
                        case=case,
                        seed=validated_seed,
                    ),
                )
                for case in cases
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


def build_sdxl_visual_model_links(
    comfy_root: Path,
    inventory: SdxlVisualInventory,
) -> ManagedSdxlVisualModelLinks:
    """Return the exact checkpoint and LoRA visibility owner."""

    return ManagedSdxlVisualModelLinks(
        model_root=resolve_active_comfy_model_root(comfy_root),
        links=(
            ManagedModelLink(
                inventory.checkpoint_source,
                "checkpoints",
                CHECKPOINT_SELECTION,
            ),
            ManagedModelLink(
                inventory.left_character.source,
                "loras",
                LEFT_CHARACTER_SELECTION,
            ),
            ManagedModelLink(
                inventory.right_character.source,
                "loras",
                RIGHT_CHARACTER_SELECTION,
            ),
            ManagedModelLink(inventory.style.source, "loras", STYLE_SELECTION),
        ),
    )


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
