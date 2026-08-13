# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Extract typed P9.4 metrics and diagnostics from Comfy history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from tools.anima_attention_coupling_workflow import (
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.comfy_api import JsonObject


@dataclass(frozen=True, slots=True)
class TextEncoderLoraHistoryEvidence:
    """Retain one successful workflow's metrics and regional diagnostics."""

    metrics: JsonObject
    diagnostics: JsonObject
    conditioning_batch_snapshot: JsonObject


def extract_text_encoder_lora_history(
    history: JsonObject,
    workflow: BuiltAnimaAttentionCouplingWorkflow,
) -> TextEncoderLoraHistoryEvidence:
    """Require one complete successful metrics and diagnostics record."""

    status = history.get("status")
    if not isinstance(status, dict) or status.get("status_str") != "success":
        raise ValueError("P9.4 history must report successful completion.")
    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("P9.4 history is missing output records.")
    metrics = _single_record(
        outputs,
        workflow.metrics_node_id,
        field="benchmark_metrics",
        label="metrics",
    )
    if metrics.get("run_id") != workflow.metrics_run_id:
        raise ValueError("P9.4 metrics identity does not match its workflow.")
    call_count = metrics.get("model_call_count")
    runtime = metrics.get("runtime_ms")
    peak = metrics.get("peak_vram_bytes")
    if not isinstance(call_count, int) or call_count < 1:
        raise ValueError("P9.4 model-call count must be positive.")
    if not isinstance(runtime, int | float) or runtime <= 0:
        raise ValueError("P9.4 instrumented runtime must be positive.")
    if not isinstance(peak, int) or peak <= 0:
        raise ValueError("P9.4 peak VRAM must be positive.")

    diagnostics = _single_record(
        outputs,
        workflow.diagnostics_node_id,
        field="regional_diagnostics",
        label="regional diagnostics",
    )
    if diagnostics.get("run_id") != workflow.diagnostics_run_id:
        raise ValueError("P9.4 diagnostics identity does not match its workflow.")
    snapshots = diagnostics.get("snapshots")
    record_count = diagnostics.get("record_count")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("P9.4 diagnostics require non-empty snapshots.")
    if record_count != len(snapshots):
        raise ValueError("P9.4 diagnostic count must match its snapshots.")
    if any(not isinstance(snapshot, dict) for snapshot in snapshots):
        raise TypeError("P9.4 diagnostic snapshots must be objects.")
    snapshot_node_id = workflow.conditioning_batch_snapshot_node_id
    snapshot_run_id = workflow.conditioning_batch_snapshot_run_id
    if snapshot_node_id is None or snapshot_run_id is None:
        raise ValueError("P9.4 workflow is missing conditioning-batch evidence.")
    conditioning_batch_snapshot = _single_record(
        outputs,
        snapshot_node_id,
        field="conditioning_batch_snapshot",
        label="conditioning-batch snapshot",
    )
    if conditioning_batch_snapshot.get("run_id") != snapshot_run_id:
        raise ValueError(
            "P9.4 conditioning-batch snapshot identity does not match its workflow."
        )
    return TextEncoderLoraHistoryEvidence(
        metrics,
        diagnostics,
        conditioning_batch_snapshot,
    )


def _single_record(
    outputs: dict[object, object],
    node_id: str,
    *,
    field: str,
    label: str,
) -> JsonObject:
    """Return one exact benchmark output record."""

    output = outputs.get(node_id)
    if not isinstance(output, dict):
        raise ValueError(f"P9.4 history is missing {label} output.")
    records = output.get(field)
    if (
        not isinstance(records, list)
        or len(records) != 1
        or not isinstance(records[0], dict)
    ):
        raise ValueError(f"P9.4 history requires one {label} record.")
    return cast(JsonObject, records[0])
