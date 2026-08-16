# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Measure zero/one/four active SDXL regional adapter uses."""

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
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_attention_coupling_integration.visual_launch import (
    sdxl_visual_sampling_launch_arguments,
)
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_attention_coupling_integration.visual_masks import (
    ManagedSdxlVisualMasks,
)
from tools.sdxl_attention_coupling_integration.visual_matrix_execution import (
    build_sdxl_visual_model_links,
)

from .scaling_cases import regional_scaling_cases
from .scaling_results import summarize_active_use_scaling
from .scaling_workflow import build_regional_scaling_steady_state_workflow
from .steady_state_runner import measure_steady_state_workflow

LOGGER = logging.getLogger(__name__)


def run_active_use_scaling(
    artifacts: IntegrationArtifacts,
    *,
    inventory: SdxlVisualInventory,
    prompts: SdxlVisualPromptSet,
    comfy_root: Path,
    repeats: int = 5,
    readiness_timeout: float = 240.0,
    prompt_timeout: float = 1200.0,
) -> Path:
    """Measure three prepared scaling profiles in one managed process."""

    if repeats < 5:
        raise ValueError("Active-use scaling requires at least five repeats.")
    declared_cases = regional_scaling_cases(
        prompts,
        left_trigger_g=inventory.left_character.prompt_g,
        left_trigger_l=inventory.left_character.prompt_l,
        right_trigger_g=inventory.right_character.prompt_g,
        right_trigger_l=inventory.right_character.prompt_l,
        style_trigger_g=inventory.style.prompt_g,
        style_trigger_l=inventory.style.prompt_l,
    )
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
            mask_names = masks.names(declared_cases[0].case.mask_profile)
            workflows = tuple(
                (
                    declared,
                    build_regional_scaling_steady_state_workflow(
                        checkpoint_name=CHECKPOINT_SELECTION,
                        mask_names=mask_names,
                        declared=declared,
                    ),
                )
                for declared in declared_cases
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
                for declared, workflow in workflows:
                    observation = measure_steady_state_workflow(
                        running.client,
                        workflow=workflow,
                        repeats=repeats,
                        first_seed=SDXL_VISUAL_SAMPLING.seed,
                        prompt_timeout=prompt_timeout,
                    )
                    observations[declared.profile.value] = asdict(observation)
                    LOGGER.info(
                        "Completed %s warmed scaling block",
                        declared.profile.value,
                    )
                port = running.port
                process = running.process
            server_cleanup = not process.is_running and is_loopback_port_available(port)
            artifacts.record_cleanup(
                process_running=process.is_running,
                port_available=is_loopback_port_available(port),
            )
    summary = summarize_active_use_scaling(observations)
    result = artifacts.root / "sdxl-active-use-scaling.json"
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
