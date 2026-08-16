# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute and persist one exact image-free SDXL visual diagnostic graph."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer

from .evidence_validation import validate_sdxl_diagnostics, validate_sdxl_metrics
from .matrix import MODES
from .visual_adapter_selections import CHECKPOINT_SELECTION
from .visual_case_model import SdxlVisualCase, VisualMode
from .visual_diagnostic_history import decode_sdxl_visual_diagnostic_history
from .visual_diagnostic_validation import (
    validate_standard_unet_diagnostics,
)
from .visual_diagnostic_workflow import build_sdxl_visual_diagnostic_workflow
from .visual_inventory import SdxlVisualInventory
from .visual_launch import sdxl_visual_sampling_launch_arguments
from .visual_masks import ManagedSdxlVisualMasks
from .visual_matrix_execution import build_sdxl_visual_model_links


def execute_sdxl_visual_diagnostics(
    artifacts: IntegrationArtifacts,
    *,
    inventory: SdxlVisualInventory,
    case: SdxlVisualCase,
    comfy_root: Path,
    readiness_timeout: float = 240.0,
    prompt_timeout: float = 1200.0,
) -> Path:
    """Run one full-mode graph and persist validated non-image evidence."""

    if not isinstance(artifacts, IntegrationArtifacts):
        raise TypeError("SDXL visual diagnostics require integration artifacts.")
    if not isinstance(inventory, SdxlVisualInventory):
        raise TypeError("SDXL visual diagnostics require external inventory.")
    if not isinstance(case, SdxlVisualCase) or case.modes != (VisualMode.FULL,):
        raise ValueError("SDXL visual diagnostics require one full-mode case.")
    links = build_sdxl_visual_model_links(comfy_root, inventory)
    masks = ManagedSdxlVisualMasks(
        input_root=comfy_root / "input",
        run_id=artifacts.run_id,
    )
    evidence: JsonObject = {}
    system_stats: JsonObject = {}
    history: JsonObject = {}
    prompt_id = ""
    wall_runtime_ms = 0.0
    server_cleanup = False
    with links:
        with masks:
            workflow = build_sdxl_visual_diagnostic_workflow(
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
                started_at = time.perf_counter()
                prompt_id = running.client.submit(workflow.prompt)
                history = running.client.wait_for_history(
                    prompt_id,
                    timeout=prompt_timeout,
                )
                wall_runtime_ms = (time.perf_counter() - started_at) * 1000.0
                decoded = decode_sdxl_visual_diagnostic_history(
                    history,
                    metrics_node_id=workflow.metrics_node_id,
                    diagnostics_node_id=workflow.diagnostics_node_id,
                )
                metrics = validate_sdxl_metrics(decoded.metrics, case.case_id)
                diagnostics = validate_sdxl_diagnostics(
                    decoded.diagnostics,
                    label=case.case_id,
                    expected_spatial_modes=MODES[0].expected_spatial_modes,
                )
                validate_standard_unet_diagnostics(diagnostics)
                if metrics["model_call_count"] != MODES[0].expected_model_calls:
                    raise ValueError(
                        "SDXL visual diagnostic model-call count is invalid."
                    )
                evidence = {"metrics": metrics, "diagnostics": diagnostics}
                port = running.port
                process = running.process
            server_cleanup = not process.is_running and is_loopback_port_available(port)
            artifacts.record_cleanup(
                process_running=process.is_running,
                port_available=is_loopback_port_available(port),
            )
            mask_evidence = masks.evidence()
    if not server_cleanup or not links.cleaned or not masks.cleaned:
        raise RuntimeError("SDXL visual diagnostic cleanup is incomplete.")
    result = artifacts.root / "sdxl-visual-diagnostics.json"
    _write_json(
        result,
        {
            "status": "completed",
            "case_id": case.case_id,
            "case_label": case.label,
            "prompt_id": prompt_id,
            "wall_runtime_ms": wall_runtime_ms,
            "workflow": workflow.prompt,
            "history": history,
            "evidence": evidence,
            "system_stats": system_stats,
            "mask_evidence": list(mask_evidence),
            "cleanup": {"server": True, "model_links": True, "masks": True},
        },
    )
    return result


def _write_json(path: Path, payload: JsonObject) -> None:
    """Atomically persist one complete image-free diagnostic result."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
