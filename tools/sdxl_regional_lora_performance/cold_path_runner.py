# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run one managed SDXL cold-path attribution matrix."""

from __future__ import annotations

import json
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

from .cold_path_measurement import measure_cold_path
from .cold_path_priming import (
    build_cold_path_primers,
    execute_cold_path_primers,
)
from .cold_path_results import summarize_cold_path
from .cold_path_workflow import build_sdxl_cold_path_workflow


def run_cold_path_attribution(
    artifacts: IntegrationArtifacts,
    *,
    inventory: SdxlVisualInventory,
    case: SdxlVisualCase,
    comfy_root: Path,
    readiness_timeout: float = 240.0,
    prompt_timeout: float = 1200.0,
) -> Path:
    """Record exactly one cold request and one warmed exact reuse."""

    links = build_sdxl_visual_model_links(comfy_root, inventory)
    masks = ManagedSdxlVisualMasks(
        input_root=comfy_root / "input",
        run_id=artifacts.run_id,
    )
    system_stats: JsonObject = {}
    server_cleanup = False
    with links:
        with masks:
            workflow = build_sdxl_cold_path_workflow(
                checkpoint_name=CHECKPOINT_SELECTION,
                mask_names=masks.names(case.mask_profile),
                case=case,
            )
            primers = build_cold_path_primers(
                checkpoint_name=CHECKPOINT_SELECTION,
                mask_names=masks.names(case.mask_profile),
                case=case,
            )
            required_node_ids = frozenset(
                {
                    *workflow.required_node_ids,
                    *(
                        node_id
                        for primer in primers
                        for node_id in primer.workflow.required_node_ids
                    ),
                }
            )
            with ManagedComfyServer(
                comfy_root=comfy_root,
                artifacts=artifacts,
                required_node_ids=required_node_ids,
                readiness_timeout=readiness_timeout,
                launch_arguments=sdxl_visual_sampling_launch_arguments(),
            ) as running:
                system_stats = running.system_stats
                primer_timings = execute_cold_path_primers(
                    running.client,
                    primers=primers,
                    seed=SDXL_VISUAL_SAMPLING.seed,
                    prompt_timeout=prompt_timeout,
                )
                observation = measure_cold_path(
                    running.client,
                    workflow=workflow,
                    first_seed=SDXL_VISUAL_SAMPLING.seed,
                    run_id=artifacts.run_id,
                    prompt_timeout=prompt_timeout,
                )
                port = running.port
                process = running.process
            server_cleanup = not process.is_running and is_loopback_port_available(port)
            artifacts.record_cleanup(
                process_running=process.is_running,
                port_available=is_loopback_port_available(port),
            )
    summary = summarize_cold_path(observation.cold, observation.warm)
    result = artifacts.root / "sdxl-cold-path-attribution.json"
    result.write_text(
        json.dumps(
            {
                "status": "completed",
                "observation": asdict(observation),
                "primers": {
                    label: asdict(timing) for label, timing in primer_timings.items()
                },
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
