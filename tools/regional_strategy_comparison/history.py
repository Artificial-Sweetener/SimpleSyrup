# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Narrow P10.2 managed history into typed evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from tools.comfy_api import ImageReference, JsonObject
from tools.comfy_integration.history_output import extract_saved_image

from .workflow import BuiltStrategyComparisonWorkflow


@dataclass(frozen=True, slots=True)
class StrategyProbeMetrics:
    """Retain exact model-call and trajectory-batch measurements."""

    runtime_ms: float
    peak_vram_bytes: int
    model_call_count: int
    model_input_batch_sizes: tuple[int, ...]
    model_input_batch_elements: int


@dataclass(frozen=True, slots=True)
class CompletedStrategyEvidence:
    """Retain one case's metrics, diagnostics, and saved image reference."""

    metrics: StrategyProbeMetrics
    diagnostics: tuple[JsonObject, ...]
    image_reference: ImageReference


def parse_completed_evidence(
    history: JsonObject,
    workflow: BuiltStrategyComparisonWorkflow,
) -> CompletedStrategyEvidence:
    """Return exact probe and diagnostic values from one completed prompt."""

    outputs = _object(history.get("outputs"), "history outputs")
    metric_output = _object(outputs.get(workflow.metrics_node_id), "metrics output")
    metric_items = _array(metric_output.get("benchmark_metrics"), "benchmark metrics")
    if len(metric_items) != 1:
        raise ValueError("P10.2 history must contain one benchmark metric record.")
    metric = _object(metric_items[0], "benchmark metric record")
    if metric.get("run_id") != workflow.metrics_run_id:
        raise ValueError("P10.2 benchmark metric identity does not match its workflow.")
    batch_sizes = tuple(
        _positive_integer(value, "model input batch size")
        for value in _array(
            metric.get("model_input_batch_sizes"),
            "model_input_batch_sizes",
        )
    )
    metrics = StrategyProbeMetrics(
        runtime_ms=_nonnegative_number(metric.get("runtime_ms"), "runtime_ms"),
        peak_vram_bytes=_nonnegative_integer(
            metric.get("peak_vram_bytes"), "peak_vram_bytes"
        ),
        model_call_count=_positive_integer(
            metric.get("model_call_count"), "model_call_count"
        ),
        model_input_batch_sizes=batch_sizes,
        model_input_batch_elements=_positive_integer(
            metric.get("model_input_batch_elements"),
            "model_input_batch_elements",
        ),
    )
    if len(batch_sizes) != metrics.model_call_count:
        raise ValueError("P10.2 must record one input batch size per model call.")
    if sum(batch_sizes) != metrics.model_input_batch_elements:
        raise ValueError("P10.2 model input batch element total is inconsistent.")
    diagnostics = _diagnostics(outputs, workflow)
    return CompletedStrategyEvidence(
        metrics,
        diagnostics,
        extract_saved_image(history, workflow.save_node_id),
    )


def _diagnostics(
    outputs: JsonObject,
    workflow: BuiltStrategyComparisonWorkflow,
) -> tuple[JsonObject, ...]:
    """Return exact Attention Coupling snapshots or an empty legacy tuple."""

    if workflow.diagnostics_node_id is None:
        if workflow.diagnostics_run_id is not None:
            raise ValueError("P10.2 legacy workflow has a diagnostics run identity.")
        return ()
    output = _object(
        outputs.get(workflow.diagnostics_node_id),
        "regional diagnostics output",
    )
    items = _array(output.get("regional_diagnostics"), "regional diagnostics")
    if len(items) != 1:
        raise ValueError("P10.2 must contain one regional diagnostics record.")
    record = _object(items[0], "regional diagnostics record")
    if record.get("run_id") != workflow.diagnostics_run_id:
        raise ValueError("P10.2 regional diagnostics identity does not match.")
    snapshots = tuple(
        _object(value, "regional diagnostic snapshot")
        for value in _array(record.get("snapshots"), "diagnostic snapshots")
    )
    if record.get("record_count") != len(snapshots):
        raise ValueError("P10.2 regional diagnostic record count is inconsistent.")
    return snapshots


def _object(value: object, label: str) -> JsonObject:
    """Narrow one JSON object."""

    if not isinstance(value, dict):
        raise TypeError(f"P10.2 {label} must be an object.")
    return cast(JsonObject, value)


def _array(value: object, label: str) -> list[object]:
    """Narrow one JSON array."""

    if not isinstance(value, list):
        raise TypeError(f"P10.2 {label} must be an array.")
    return value


def _positive_integer(value: object, label: str) -> int:
    """Narrow one positive integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"P10.2 {label} must be a positive integer.")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    """Narrow one nonnegative integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"P10.2 {label} must be a nonnegative integer.")
    return value


def _nonnegative_number(value: object, label: str) -> float:
    """Narrow one finite nonnegative number."""

    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError(f"P10.2 {label} must be a nonnegative number.")
    result = float(value)
    if result != result or result == float("inf"):
        raise ValueError(f"P10.2 {label} must be finite.")
    return result
