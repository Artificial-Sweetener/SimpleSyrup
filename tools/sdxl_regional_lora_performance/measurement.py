# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decode and summarize matched SDXL regional-LoRA timing evidence."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import cast

from tools.comfy_api import JsonObject


@dataclass(frozen=True, slots=True)
class SdxlRegionalLoraTiming:
    """Retain one CUDA-synchronized sampler observation."""

    runtime_ms: float
    model_call_count: int
    model_input_batch_elements: int
    model_input_batch_sizes: tuple[int, ...]
    peak_vram_bytes: int


def decode_timing(history: JsonObject, metrics_node_id: str) -> SdxlRegionalLoraTiming:
    """Decode one exact benchmark metric object from Comfy history."""

    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("SDXL performance history is missing outputs.")
    output = outputs.get(metrics_node_id)
    if not isinstance(output, dict):
        raise ValueError("SDXL performance history is missing its metric node.")
    values = output.get("benchmark_metrics")
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError("SDXL performance metrics must contain one observation.")
    metrics = values[0]
    if not isinstance(metrics, dict):
        raise ValueError("SDXL performance observation must be an object.")
    sizes = metrics.get("model_input_batch_sizes")
    if not isinstance(sizes, list) or not all(
        isinstance(value, int) for value in sizes
    ):
        raise ValueError("SDXL performance batch sizes are invalid.")
    return SdxlRegionalLoraTiming(
        runtime_ms=float(metrics["runtime_ms"]),
        model_call_count=int(metrics["model_call_count"]),
        model_input_batch_elements=int(metrics["model_input_batch_elements"]),
        model_input_batch_sizes=tuple(cast(list[int], sizes)),
        peak_vram_bytes=int(metrics["peak_vram_bytes"]),
    )


def median_runtime_ms(values: tuple[SdxlRegionalLoraTiming, ...]) -> float:
    """Return the median runtime from at least three measured repeats."""

    if len(values) < 3:
        raise ValueError("SDXL performance comparison requires three repeats.")
    return float(statistics.median(value.runtime_ms for value in values))
