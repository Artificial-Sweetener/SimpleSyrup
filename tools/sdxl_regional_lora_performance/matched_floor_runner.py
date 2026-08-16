# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Measure the four-mode warmed SDXL regional execution floor."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.sdxl_attention_coupling_integration.sampling_controls import (
    SDXL_VISUAL_SAMPLING,
)
from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    CHECKPOINT_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import SdxlVisualCase
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_attention_coupling_integration.visual_launch import (
    sdxl_visual_sampling_launch_arguments,
)
from tools.sdxl_attention_coupling_integration.visual_masks import (
    ManagedSdxlVisualMasks,
)
from tools.sdxl_attention_coupling_integration.visual_matrix_execution import (
    build_sdxl_visual_model_links,
)

from .conventional_variant_workflow import (
    ConventionalVariantBranch,
    build_conventional_variant_steady_state_workflow,
)
from .matched_floor_results import summarize_matched_floor
from .steady_state_runner import measure_steady_state_workflow
from .steady_state_workflow import BuiltSdxlSteadyStateWorkflow
from .two_adapter_workflow import (
    TwoAdapterPerformanceMode,
    build_two_adapter_steady_state_workflow,
)

LOGGER = logging.getLogger(__name__)


def run_matched_floor_comparison(
    artifacts: IntegrationArtifacts,
    *,
    inventory: SdxlVisualInventory,
    case: SdxlVisualCase,
    comfy_root: Path,
    repeats: int = 5,
    readiness_timeout: float = 240.0,
    prompt_timeout: float = 1200.0,
) -> Path:
    """Measure four prepared modes in one managed Comfy process."""

    if repeats < 5:
        raise ValueError("Matched floor comparison requires at least five repeats.")
    links = build_sdxl_visual_model_links(comfy_root, inventory)
    masks = ManagedSdxlVisualMasks(
        input_root=comfy_root / "input",
        run_id=artifacts.run_id,
    )
    observations: dict[str, dict[str, object]] = {}
    system_stats: JsonObject = {}
    server_cleanup = False
    with links:
        with masks:
            workflows = _build_workflows(
                case=case,
                mask_names=masks.names(case.mask_profile),
            )
            required = frozenset(
                node_id
                for _, workflow in workflows
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
                for label, workflow in workflows:
                    observation = measure_steady_state_workflow(
                        running.client,
                        workflow=workflow,
                        repeats=repeats,
                        first_seed=SDXL_VISUAL_SAMPLING.seed,
                        prompt_timeout=prompt_timeout,
                    )
                    observations[label] = asdict(observation)
                    LOGGER.info("Completed %s warmed timing block", label)
                port = running.port
                process = running.process
            server_cleanup = not process.is_running and is_loopback_port_available(port)
            artifacts.record_cleanup(
                process_running=process.is_running,
                port_available=is_loopback_port_available(port),
            )
    summary = summarize_matched_floor(observations)
    result = artifacts.root / "sdxl-matched-floor.json"
    result.write_text(
        json.dumps(
            {
                "status": "completed",
                "repeats": repeats,
                "observations": observations,
                "summary": asdict(summary),
                "system_stats": system_stats,
                "cleanup": {
                    "server": server_cleanup,
                    "model_links": links.cleaned,
                    "masks": masks.cleaned,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _build_workflows(
    *,
    case: SdxlVisualCase,
    mask_names: tuple[str, str],
) -> tuple[tuple[str, BuiltSdxlSteadyStateWorkflow], ...]:
    """Build the predeclared global, floor, and regional mode order."""

    global_workflow = build_two_adapter_steady_state_workflow(
        checkpoint_name=CHECKPOINT_SELECTION,
        mask_names=mask_names,
        case=case,
        mode=TwoAdapterPerformanceMode.GLOBAL_REFERENCE,
    )
    left_workflow = build_conventional_variant_steady_state_workflow(
        checkpoint_name=CHECKPOINT_SELECTION,
        case=case,
        branch=ConventionalVariantBranch.LEFT,
    )
    right_workflow = build_conventional_variant_steady_state_workflow(
        checkpoint_name=CHECKPOINT_SELECTION,
        case=case,
        branch=ConventionalVariantBranch.RIGHT,
    )
    regional_workflow = build_two_adapter_steady_state_workflow(
        checkpoint_name=CHECKPOINT_SELECTION,
        mask_names=mask_names,
        case=case,
        mode=TwoAdapterPerformanceMode.REGIONAL,
    )
    return (
        (TwoAdapterPerformanceMode.GLOBAL_REFERENCE.value, global_workflow),
        (ConventionalVariantBranch.LEFT.value, left_workflow),
        (ConventionalVariantBranch.RIGHT.value, right_workflow),
        (TwoAdapterPerformanceMode.REGIONAL.value, regional_workflow),
    )
