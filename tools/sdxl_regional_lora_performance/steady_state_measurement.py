# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decode cache-realistic synchronized SDXL completion timings."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from tools.comfy_api import JsonObject


@dataclass(frozen=True, slots=True)
class SdxlSteadyStateTiming:
    """Retain one seed-only prepared-graph timing observation."""

    runtime_ms: float
    seed: int


def decode_steady_state_timing(
    history: JsonObject,
    *,
    completion_node_id: str,
    started_at_ns: int,
    seed: int,
) -> SdxlSteadyStateTiming:
    """Decode a synchronized completion timestamp and calculate wall time."""

    if isinstance(started_at_ns, bool) or started_at_ns < 1:
        raise ValueError("Steady-state start timestamp must be positive.")
    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Steady-state history is missing outputs.")
    output = outputs.get(completion_node_id)
    if not isinstance(output, dict):
        raise ValueError("Steady-state history is missing its completion node.")
    values = output.get("benchmark_completion")
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError("Steady-state completion must contain one observation.")
    completion = values[0]
    if not isinstance(completion, dict):
        raise ValueError("Steady-state completion observation must be an object.")
    completed_at_ns = completion.get("completed_at_ns")
    if isinstance(completed_at_ns, bool) or not isinstance(completed_at_ns, int):
        raise ValueError("Steady-state completion timestamp is invalid.")
    elapsed_ns = completed_at_ns - started_at_ns
    if elapsed_ns <= 0:
        raise ValueError("Steady-state completion must follow submission start.")
    return SdxlSteadyStateTiming(runtime_ms=elapsed_ns / 1_000_000.0, seed=seed)


def median_steady_state_runtime_ms(
    values: tuple[SdxlSteadyStateTiming, ...],
) -> float:
    """Return the median runtime from at least three seed-only repeats."""

    if len(values) < 3:
        raise ValueError("Steady-state comparison requires three repeats.")
    return float(statistics.median(value.runtime_ms for value in values))
