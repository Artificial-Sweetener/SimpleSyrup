# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run one managed sampler-free materialization parity comparison."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer
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

from .results import decode_materialization_parity
from .workflow import build_materialization_parity_workflow


def run_materialization_parity(
    artifacts: IntegrationArtifacts,
    *,
    inventory: SdxlVisualInventory,
    case: SdxlVisualCase,
    comfy_root: Path,
    readiness_timeout: float = 240.0,
    prompt_timeout: float = 1200.0,
) -> Path:
    """Persist one complete comparison and managed lifecycle evidence."""

    links = build_sdxl_visual_model_links(comfy_root, inventory)
    masks = ManagedSdxlVisualMasks(
        input_root=comfy_root / "input",
        run_id=artifacts.run_id,
    )
    system_stats: JsonObject = {}
    server_cleanup = False
    started_at_ns = 0
    completed_at_ns = 0
    with links:
        with masks:
            workflow = build_materialization_parity_workflow(
                run_id=artifacts.run_id,
                checkpoint_name=CHECKPOINT_SELECTION,
                mask_names=masks.names(case.mask_profile),
                case=case,
            )
            with ManagedComfyServer(
                comfy_root=comfy_root,
                artifacts=artifacts,
                required_node_ids=workflow.required_node_ids,
                readiness_timeout=readiness_timeout,
                launch_arguments=sdxl_visual_sampling_launch_arguments(),
            ) as running:
                system_stats = running.system_stats
                started_at_ns = time.perf_counter_ns()
                prompt_id = running.client.submit(workflow.prompt)
                history = running.client.wait_for_history(
                    prompt_id,
                    timeout=prompt_timeout,
                )
                completed_at_ns = time.perf_counter_ns()
                comparison = decode_materialization_parity(
                    history,
                    terminal_node_id=workflow.terminal_node_id,
                    expected_run_id=artifacts.run_id,
                )
                port = running.port
                process = running.process
            server_cleanup = not process.is_running and is_loopback_port_available(port)
            artifacts.record_cleanup(
                process_running=process.is_running,
                port_available=is_loopback_port_available(port),
            )
    result = artifacts.root / "materialization-parity.json"
    result.write_text(
        json.dumps(
            {
                "status": "completed",
                "workflow_elapsed_ms": (completed_at_ns - started_at_ns) / 1_000_000.0,
                "comparison": comparison,
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
