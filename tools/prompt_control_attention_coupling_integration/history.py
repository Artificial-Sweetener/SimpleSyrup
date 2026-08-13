# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decode exact P9.1 evidence records from terminal Comfy history."""

from __future__ import annotations

from dataclasses import dataclass

from tools.anima_attention_coupling_workflow import (
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.comfy_api import JsonObject


@dataclass(frozen=True, slots=True)
class PromptControlAttentionOutputs:
    """Retain one case's conditioning, metric, and diagnostic evidence."""

    expansion: JsonObject
    snapshot: JsonObject
    metrics: JsonObject
    diagnostics: JsonObject


def parse_outputs(
    history: JsonObject,
    workflow: BuiltAnimaAttentionCouplingWorkflow,
) -> PromptControlAttentionOutputs:
    """Return exact records after requiring one successful terminal history."""

    status = _object(history.get("status"), "history.status")
    if status.get("status_str") != "success":
        raise RuntimeError(f"P9.1 Comfy execution failed: {status.get('messages')!r}.")
    outputs = _object(history.get("outputs"), "history.outputs")
    expansion_id = workflow.conditioning_evidence.expansion_node_id
    snapshot_id = workflow.conditioning_evidence.snapshot_node_id
    if expansion_id is None or snapshot_id is None:
        raise ValueError("P9.1 workflow is missing conditioning evidence nodes.")
    expansion = _single(
        outputs,
        expansion_id,
        "prompt_control_expansion",
    )
    snapshot = _single(
        outputs,
        snapshot_id,
        "prompt_control_snapshot",
    )
    metrics = _single(
        outputs,
        workflow.metrics_node_id,
        "benchmark_metrics",
    )
    diagnostics = _single(
        outputs,
        workflow.diagnostics_node_id,
        "regional_diagnostics",
    )
    return PromptControlAttentionOutputs(expansion, snapshot, metrics, diagnostics)


def _single(
    outputs: JsonObject,
    node_id: str,
    field: str,
) -> JsonObject:
    """Return one JSON object from one exact output-node list."""

    output = _object(outputs.get(node_id), f"node {node_id}")
    values = output.get(field)
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"P9.1 {field} must contain exactly one record.")
    return _object(values[0], field)


def _object(value: object, field: str) -> JsonObject:
    """Narrow one JSON object with string keys."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"P9.1 {field} must be an object.")
    return value
