# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run matched native and regional SDXL LoRA steady-state blocks."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from tools.comfy_api import JsonObject, LoopbackComfyClient
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    CHECKPOINT_SELECTION,
    LEFT_CHARACTER_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_attention_coupling_integration.visual_launch import (
    sdxl_visual_sampling_launch_arguments,
)
from tools.sdxl_attention_coupling_integration.visual_matrix_execution import (
    build_sdxl_visual_model_links,
)
from tools.sdxl_full_strength_lora_fidelity.cases import (
    CharacterFidelitySpec,
    FidelityExecutionMode,
    FullStrengthFidelityCase,
    fidelity_cases,
)
from tools.sdxl_full_strength_lora_fidelity.mask import ManagedAllOneMask

from .measurement import SdxlRegionalLoraTiming, decode_timing, median_runtime_ms
from .workflow import build_performance_workflow

LOGGER = logging.getLogger(__name__)


def run_steady_state_comparison(
    artifacts: IntegrationArtifacts,
    *,
    inventory: SdxlVisualInventory,
    comfy_root: Path,
    base_positive_g: str,
    base_positive_l: str,
    negative_g: str,
    negative_l: str,
    repeats: int = 3,
    readiness_timeout: float = 240.0,
    prompt_timeout: float = 1200.0,
) -> Path:
    """Measure consecutive warmed blocks in one managed Comfy process."""

    if repeats < 3:
        raise ValueError("SDXL steady-state comparison requires three repeats.")
    character = CharacterFidelitySpec(
        "matched-character",
        inventory.left_character.label,
        LEFT_CHARACTER_SELECTION,
        inventory.left_character.prompt_g,
        inventory.left_character.prompt_l,
    )
    cases = fidelity_cases((character,))
    links = build_sdxl_visual_model_links(comfy_root, inventory)
    mask = ManagedAllOneMask(input_root=comfy_root / "input", run_id=artifacts.run_id)
    observations: dict[str, dict[str, object]] = {}
    server_cleanup = False
    system_stats: JsonObject = {}
    with links:
        with mask:
            preview = tuple(
                build_performance_workflow(
                    run_id=f"{artifacts.run_id}:preview:{case.case_id}",
                    checkpoint_name=CHECKPOINT_SELECTION,
                    mask_name=mask.name,
                    base_positive_g=base_positive_g,
                    base_positive_l=base_positive_l,
                    negative_g=negative_g,
                    negative_l=negative_l,
                    case=case,
                )
                for case in cases
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
                for case in cases:
                    mode = case.mode.value
                    warmup = _execute_once(
                        running.client,
                        run_id=f"{artifacts.run_id}:{mode}:warmup",
                        case=case,
                        inventory=inventory,
                        mask_name=mask.name,
                        base_positive_g=base_positive_g,
                        base_positive_l=base_positive_l,
                        negative_g=negative_g,
                        negative_l=negative_l,
                        prompt_timeout=prompt_timeout,
                    )
                    measured = tuple(
                        _execute_once(
                            running.client,
                            run_id=f"{artifacts.run_id}:{mode}:repeat-{index + 1}",
                            case=case,
                            inventory=inventory,
                            mask_name=mask.name,
                            base_positive_g=base_positive_g,
                            base_positive_l=base_positive_l,
                            negative_g=negative_g,
                            negative_l=negative_l,
                            prompt_timeout=prompt_timeout,
                        )
                        for index in range(repeats)
                    )
                    observations[mode] = {
                        "warmup": asdict(warmup),
                        "measured": [asdict(value) for value in measured],
                        "median_runtime_ms": median_runtime_ms(measured),
                    }
                    LOGGER.info("Completed %s steady-state block", mode)
                port = running.port
                process = running.process
            server_cleanup = not process.is_running and is_loopback_port_available(port)
            artifacts.record_cleanup(
                process_running=process.is_running,
                port_available=is_loopback_port_available(port),
            )
    native = _recorded_median(
        observations[FidelityExecutionMode.GLOBAL_REFERENCE.value]
    )
    regional = _recorded_median(
        observations[FidelityExecutionMode.REGIONAL_ALL_ONE.value]
    )
    result = artifacts.root / "sdxl-regional-lora-steady-state.json"
    result.write_text(
        json.dumps(
            {
                "status": "completed",
                "repeats": repeats,
                "observations": observations,
                "regional_to_native_ratio": regional / native,
                "regional_overhead_percent": ((regional / native) - 1.0) * 100.0,
                "gate_ratio": 2.93,
                "gate_passed": regional / native <= 2.93,
                "system_stats": system_stats,
                "cleanup": {
                    "server": server_cleanup,
                    "model_links": links.cleaned,
                    "mask": mask.cleaned,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _execute_once(
    client: LoopbackComfyClient,
    *,
    run_id: str,
    case: FullStrengthFidelityCase,
    inventory: SdxlVisualInventory,
    mask_name: str,
    base_positive_g: str,
    base_positive_l: str,
    negative_g: str,
    negative_l: str,
    prompt_timeout: float,
) -> SdxlRegionalLoraTiming:
    """Submit one unique graph and decode its synchronized timing."""

    if not isinstance(client, LoopbackComfyClient):
        raise TypeError("SDXL performance execution requires a loopback client.")
    if not isinstance(case, FullStrengthFidelityCase):
        raise TypeError("SDXL performance execution requires a fidelity case.")
    del inventory
    workflow = build_performance_workflow(
        run_id=run_id,
        checkpoint_name=CHECKPOINT_SELECTION,
        mask_name=mask_name,
        base_positive_g=base_positive_g,
        base_positive_l=base_positive_l,
        negative_g=negative_g,
        negative_l=negative_l,
        case=case,
    )
    prompt_id = client.submit(workflow.prompt)
    history = client.wait_for_history(prompt_id, timeout=prompt_timeout)
    return decode_timing(history, workflow.metrics_node_id)


def _recorded_median(observation: dict[str, object]) -> float:
    """Narrow one internally recorded median at the serialization boundary."""

    value = observation.get("median_runtime_ms")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("SDXL performance median must be numeric.")
    return float(value)
