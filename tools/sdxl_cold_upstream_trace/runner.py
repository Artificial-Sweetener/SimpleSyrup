# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run one managed node-level trace of the unchanged cold regional graph."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.execution_trace import LOOPBACK_EXECUTION_TRACE
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
from tools.sdxl_regional_lora_performance.cold_path_priming import (
    build_cold_path_primers,
    execute_cold_path_primers,
)
from tools.sdxl_regional_lora_performance.cold_path_results import (
    decode_cold_path_timing,
)
from tools.sdxl_regional_lora_performance.cold_path_workflow import (
    BuiltSdxlColdPathWorkflow,
)

from .profile_workflow import build_profiled_cold_path_workflow
from .results import attribute_cold_upstream


def run_cold_upstream_trace(
    artifacts: IntegrationArtifacts,
    *,
    inventory: SdxlVisualInventory,
    case: SdxlVisualCase,
    comfy_root: Path,
    readiness_timeout: float = 240.0,
    prompt_timeout: float = 1200.0,
) -> Path:
    """Persist one node-level trace plus the existing cold-stage diagnostics."""

    links = build_sdxl_visual_model_links(comfy_root, inventory)
    masks = ManagedSdxlVisualMasks(
        input_root=comfy_root / "input",
        run_id=artifacts.run_id,
    )
    system_stats: JsonObject = {}
    server_cleanup = False
    with links:
        with masks:
            workflow: BuiltSdxlColdPathWorkflow = build_profiled_cold_path_workflow(
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
                execution_id = f"{artifacts.run_id}:cold-upstream"
                prompt = workflow.prompt_for_execution(
                    seed=SDXL_VISUAL_SAMPLING.seed,
                    run_id=execution_id,
                )
                trace = LOOPBACK_EXECUTION_TRACE.execute(
                    running.client,
                    prompt=prompt,
                    timeout=prompt_timeout,
                )
                cold = decode_cold_path_timing(
                    trace.history,
                    terminal_node_id=workflow.terminal_node_id,
                    started_at_ns=trace.submission_started_at_ns,
                    seed=SDXL_VISUAL_SAMPLING.seed,
                )
                attribution = attribute_cold_upstream(
                    trace,
                    cold,
                    sampler_node_id=workflow.sampler_node_id,
                )
                port = running.port
                process = running.process
            server_cleanup = not process.is_running and is_loopback_port_available(port)
            artifacts.record_cleanup(
                process_running=process.is_running,
                port_available=is_loopback_port_available(port),
            )
    result = artifacts.root / "sdxl-cold-upstream-trace.json"
    result.write_text(
        json.dumps(
            {
                "status": "completed",
                "cold": asdict(cold),
                "trace": {
                    "elapsed_ms": trace.elapsed_ms,
                    "cached_node_ids": trace.cached_node_ids,
                    "nodes": [asdict(node) for node in trace.nodes],
                },
                "attribution": asdict(attribution),
                "primers": {
                    label: asdict(timing) for label, timing in primer_timings.items()
                },
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
