# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decode and validate standard-UNet cold-attribution evidence."""

from __future__ import annotations

from dataclasses import dataclass

from tools.comfy_api import JsonObject

_REQUIRED_COLD_STAGES = frozenset(
    {
        "admission_resolution",
        "variant_materialization",
        "variant_shell",
        "template_preparation",
        "model_residency",
        "sampling",
    }
)
_TOP_LEVEL_STAGES = (
    "admission_resolution",
    "template_preparation",
    "model_residency",
    "sampling",
)


@dataclass(frozen=True, slots=True)
class ColdPathStageTiming:
    """Retain one ordered structured stage observation."""

    stage: str
    elapsed_ms: float
    metadata: JsonObject


@dataclass(frozen=True, slots=True)
class SdxlColdPathTiming:
    """Retain one synchronized cold or warmed workflow execution."""

    run_id: str
    seed: int
    runtime_ms: float
    stages: tuple[ColdPathStageTiming, ...]
    model_call_count: int
    peak_vram_bytes: int


@dataclass(frozen=True, slots=True)
class SdxlColdPathSummary:
    """Attribute the cold interval without double-counting nested stages."""

    cold_runtime_ms: float
    warm_runtime_ms: float
    stage_totals_ms: dict[str, float]
    top_level_accounted_ms: float
    unattributed_ms: float
    materialized_parameter_count: int
    materialized_parameter_bytes: int


def decode_cold_path_timing(
    history: JsonObject,
    *,
    terminal_node_id: str,
    started_at_ns: int,
    seed: int,
) -> SdxlColdPathTiming:
    """Decode one synchronized benchmark terminal and its ordered stages."""

    if isinstance(started_at_ns, bool) or not isinstance(started_at_ns, int):
        raise TypeError("Cold-path start timestamp must be an integer.")
    if started_at_ns < 1:
        raise ValueError("Cold-path start timestamp must be positive.")
    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Cold-path history is missing outputs.")
    output = outputs.get(terminal_node_id)
    if not isinstance(output, dict):
        raise ValueError("Cold-path history is missing its terminal node.")
    values = output.get("cold_path_diagnostics")
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError("Cold-path terminal must contain one diagnostic object.")
    payload = values[0]
    if not isinstance(payload, dict):
        raise ValueError("Cold-path terminal diagnostic must be an object.")
    run_id = payload.get("run_id")
    completed_at_ns = payload.get("completed_at_ns")
    records = payload.get("records")
    model_call_count = payload.get("model_call_count")
    peak_vram_bytes = payload.get("peak_vram_bytes")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Cold-path terminal run id is invalid.")
    if isinstance(completed_at_ns, bool) or not isinstance(completed_at_ns, int):
        raise ValueError("Cold-path completion timestamp is invalid.")
    if completed_at_ns <= started_at_ns:
        raise ValueError("Cold-path completion must follow submission start.")
    if not isinstance(records, list) or not records:
        raise ValueError("Cold-path terminal contains no stage records.")
    if isinstance(model_call_count, bool) or not isinstance(model_call_count, int):
        raise ValueError("Cold-path model-call count is invalid.")
    if isinstance(peak_vram_bytes, bool) or not isinstance(peak_vram_bytes, int):
        raise ValueError("Cold-path peak allocation is invalid.")
    stages = tuple(_decode_stage(record) for record in records)
    return SdxlColdPathTiming(
        run_id=run_id,
        seed=seed,
        runtime_ms=(completed_at_ns - started_at_ns) / 1_000_000.0,
        stages=stages,
        model_call_count=model_call_count,
        peak_vram_bytes=peak_vram_bytes,
    )


def summarize_cold_path(
    cold: SdxlColdPathTiming,
    warm: SdxlColdPathTiming,
) -> SdxlColdPathSummary:
    """Require the complete matrix and calculate non-overlapping attribution."""

    cold_stages = {stage.stage for stage in cold.stages}
    missing = _REQUIRED_COLD_STAGES - cold_stages
    if missing:
        raise ValueError(
            f"Cold-path attribution is missing stages: {sorted(missing)!r}."
        )
    warm_stages = {stage.stage for stage in warm.stages}
    if not {"model_residency", "sampling"}.issubset(warm_stages):
        raise ValueError("Warmed attribution requires residency and sampling stages.")
    if cold.model_call_count != 30 or warm.model_call_count != 30:
        raise ValueError("Cold-path attribution requires exactly 30 model calls.")
    totals: dict[str, float] = {}
    for stage in cold.stages:
        totals[stage.stage] = totals.get(stage.stage, 0.0) + stage.elapsed_ms
    accounted = sum(totals[stage] for stage in _TOP_LEVEL_STAGES)
    materialized = tuple(
        stage for stage in cold.stages if stage.stage == "variant_materialization"
    )
    return SdxlColdPathSummary(
        cold_runtime_ms=cold.runtime_ms,
        warm_runtime_ms=warm.runtime_ms,
        stage_totals_ms=totals,
        top_level_accounted_ms=accounted,
        unattributed_ms=cold.runtime_ms - accounted,
        materialized_parameter_count=sum(
            _metadata_int(stage, "parameter_count") for stage in materialized
        ),
        materialized_parameter_bytes=sum(
            _metadata_int(stage, "parameter_bytes") for stage in materialized
        ),
    )


def _decode_stage(value: object) -> ColdPathStageTiming:
    """Narrow one JSON stage without discarding its bounded metadata."""

    if not isinstance(value, dict):
        raise ValueError("Cold-path stage must be an object.")
    stage = value.get("stage")
    elapsed_ms = value.get("elapsed_ms")
    if not isinstance(stage, str) or not stage:
        raise ValueError("Cold-path stage name is invalid.")
    if isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, int | float):
        raise ValueError("Cold-path stage elapsed time is invalid.")
    if float(elapsed_ms) < 0.0:
        raise ValueError("Cold-path stage elapsed time cannot be negative.")
    metadata = {
        str(key): item
        for key, item in value.items()
        if key not in {"stage", "elapsed_ms"}
    }
    return ColdPathStageTiming(stage, float(elapsed_ms), metadata)


def _metadata_int(stage: ColdPathStageTiming, key: str) -> int:
    """Return one required non-negative integer stage metric."""

    value = stage.metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Cold-path stage metric {key!r} is invalid.")
    return value
