# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Measure warmed vanilla and two-region SDXL LoRA execution in one process."""

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

from .steady_state_runner import measure_steady_state_workflow
from .two_adapter_workflow import (
    TwoAdapterPerformanceMode,
    build_two_adapter_steady_state_workflow,
)

LOGGER = logging.getLogger(__name__)


def run_two_adapter_comparison(
    artifacts: IntegrationArtifacts,
    *,
    inventory: SdxlVisualInventory,
    case: SdxlVisualCase,
    comfy_root: Path,
    repeats: int = 5,
    readiness_timeout: float = 240.0,
    prompt_timeout: float = 1200.0,
) -> Path:
    """Measure adjacent warmed blocks with the same two adapter payloads."""

    if repeats < 5:
        raise ValueError("Two-adapter comparison requires at least five repeats.")
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
            mask_names = masks.names(case.mask_profile)
            preview = tuple(
                build_two_adapter_steady_state_workflow(
                    checkpoint_name=CHECKPOINT_SELECTION,
                    mask_names=mask_names,
                    case=case,
                    mode=mode,
                )
                for mode in TwoAdapterPerformanceMode
            )
            required = frozenset(
                node_id
                for workflow in preview
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
                for mode in TwoAdapterPerformanceMode:
                    workflow = build_two_adapter_steady_state_workflow(
                        checkpoint_name=CHECKPOINT_SELECTION,
                        mask_names=mask_names,
                        case=case,
                        mode=mode,
                    )
                    first_seed = SDXL_VISUAL_SAMPLING.seed
                    observation = measure_steady_state_workflow(
                        running.client,
                        workflow=workflow,
                        repeats=repeats,
                        first_seed=first_seed,
                        prompt_timeout=prompt_timeout,
                    )
                    observations[mode.value] = asdict(observation)
                    LOGGER.info("Completed %s warmed timing block", mode.value)
                port = running.port
                process = running.process
            server_cleanup = not process.is_running and is_loopback_port_available(port)
            artifacts.record_cleanup(
                process_running=process.is_running,
                port_available=is_loopback_port_available(port),
            )
    vanilla = _recorded_median(
        observations[TwoAdapterPerformanceMode.GLOBAL_REFERENCE.value]
    )
    regional = _recorded_median(observations[TwoAdapterPerformanceMode.REGIONAL.value])
    result = artifacts.root / "sdxl-two-adapter-steady-state.json"
    result.write_text(
        json.dumps(
            {
                "status": "completed",
                "repeats": repeats,
                "observations": observations,
                "regional_to_vanilla_ratio": regional / vanilla,
                "regional_overhead_percent": ((regional / vanilla) - 1.0) * 100.0,
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


def _recorded_median(observation: dict[str, object]) -> float:
    """Narrow one internally recorded median at the JSON boundary."""

    value = observation.get("median_runtime_ms")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("Two-adapter performance median must be numeric.")
    return float(value)
