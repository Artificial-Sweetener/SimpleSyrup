# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate managed SDXL image, runtime, and regional diagnostic evidence."""

from __future__ import annotations

import math
from io import BytesIO
from typing import cast

from PIL import Image

from tools.comfy_api import JsonObject


def validate_sdxl_metrics(metrics: JsonObject, label: str) -> JsonObject:
    """Require exact positive runtime, VRAM, and model-call measurements."""

    model_calls = metrics.get("model_call_count")
    runtime_ms = metrics.get("runtime_ms")
    peak_vram = metrics.get("peak_vram_bytes")
    if isinstance(model_calls, bool) or not isinstance(model_calls, int):
        raise TypeError(f"SDXL {label} model-call count must be an integer.")
    if not isinstance(runtime_ms, int | float) or float(runtime_ms) <= 0:
        raise ValueError(f"SDXL {label} runtime must be positive.")
    if isinstance(peak_vram, bool) or not isinstance(peak_vram, int) or peak_vram <= 0:
        raise ValueError(f"SDXL {label} peak VRAM must be positive.")
    return metrics


def validate_sdxl_diagnostics(
    diagnostics: JsonObject,
    *,
    label: str,
    expected_spatial_modes: frozenset[str],
) -> JsonObject:
    """Require backend, strategy, invariant denoiser work, and spatial coverage."""

    record_count = diagnostics.get("record_count")
    snapshots = diagnostics.get("snapshots")
    if isinstance(record_count, bool) or not isinstance(record_count, int):
        raise TypeError(f"SDXL {label} diagnostic count must be an integer.")
    if record_count < 1 or not isinstance(snapshots, list) or not snapshots:
        return _validate_composition_diagnostics(
            diagnostics,
            label=label,
            expected_spatial_modes=expected_spatial_modes,
        )
    observed_modes: set[str] = set()
    for value in snapshots:
        if not isinstance(value, dict):
            raise TypeError(f"SDXL {label} diagnostic snapshot must be an object.")
        snapshot = cast(JsonObject, value)
        if snapshot.get("strategy") != "attention_coupling":
            raise ValueError(f"SDXL {label} diagnostic strategy is invalid.")
        backend = snapshot.get("backend")
        if not isinstance(backend, str) or not backend.endswith(".UNetModel"):
            raise ValueError(f"SDXL {label} diagnostic backend is not UNetModel.")
        estimated_work = snapshot.get("estimated_work")
        if not isinstance(estimated_work, dict):
            raise TypeError(f"SDXL {label} diagnostic work estimate is invalid.")
        work = cast(JsonObject, estimated_work)
        if work.get("denoiser_call_multiplier") != 1.0:
            raise ValueError(f"SDXL {label} duplicated denoiser execution.")
        active_regions = snapshot.get("active_region_indices")
        branch_multiplier = work.get("cross_attention_branch_multiplier")
        if not isinstance(active_regions, list) or branch_multiplier != float(
            len(active_regions) + 1
        ):
            raise ValueError(
                f"SDXL {label} cross-attention branch estimate is invalid."
            )
        spatial_mode = snapshot.get("spatial_mode")
        if isinstance(spatial_mode, str):
            observed_modes.add(spatial_mode)
    if not expected_spatial_modes <= observed_modes:
        raise ValueError(
            f"SDXL {label} diagnostics missing spatial modes "
            f"{sorted(expected_spatial_modes - observed_modes)!r}."
        )
    return diagnostics


def _validate_composition_diagnostics(
    diagnostics: JsonObject,
    *,
    label: str,
    expected_spatial_modes: frozenset[str],
) -> JsonObject:
    """Validate persistent-variant composition evidence for a full-latent run."""

    if expected_spatial_modes != frozenset({"full"}):
        raise ValueError(
            f"SDXL {label} diagnostics must contain spatial-mode snapshots."
        )
    record_count = diagnostics.get("composition_record_count")
    values = diagnostics.get("composition")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count < 1
        or not isinstance(values, list)
        or not values
    ):
        raise ValueError(f"SDXL {label} diagnostics must contain snapshots.")
    for value in values:
        if not isinstance(value, dict):
            raise TypeError(f"SDXL {label} composition diagnostic must be an object.")
        stage = value.get("stage")
        progress = value.get("denoising_progress")
        sampling_sigma = value.get("sampling_sigma")
        multipliers = value.get("schedule_multipliers")
        schedules = value.get("adapter_schedules")
        if stage not in {"composition", "specialization", "consolidation"}:
            raise ValueError(f"SDXL {label} composition stage is invalid.")
        if (
            isinstance(progress, bool)
            or not isinstance(progress, int | float)
            or not math.isfinite(float(progress))
            or not 0.0 <= float(progress) <= 1.0
        ):
            raise ValueError(f"SDXL {label} composition progress is invalid.")
        if not isinstance(multipliers, list) or not multipliers:
            raise ValueError(f"SDXL {label} composition schedule is invalid.")
        if any(
            isinstance(multiplier, bool)
            or not isinstance(multiplier, int | float)
            or not math.isfinite(float(multiplier))
            for multiplier in multipliers
        ):
            raise ValueError(f"SDXL {label} composition multiplier is invalid.")
        if (
            isinstance(sampling_sigma, bool)
            or not isinstance(sampling_sigma, int | float)
            or not math.isfinite(float(sampling_sigma))
            or float(sampling_sigma) < 0.0
        ):
            raise ValueError(f"SDXL {label} composition sigma is invalid.")
        _validate_adapter_schedules(schedules, len(multipliers), label)
    return diagnostics


def _validate_adapter_schedules(
    schedules: object,
    expected_count: int,
    label: str,
) -> None:
    """Require exact converted schedule-boundary evidence per multiplier."""

    if not isinstance(schedules, list) or len(schedules) != expected_count:
        raise ValueError(f"SDXL {label} adapter schedules are invalid.")
    for schedule in schedules:
        if not isinstance(schedule, list) or not schedule:
            raise ValueError(f"SDXL {label} adapter schedule is invalid.")
        for boundary in schedule:
            if not isinstance(boundary, dict):
                raise TypeError(f"SDXL {label} schedule boundary must be an object.")
            _validate_schedule_boundary(cast(JsonObject, boundary), label)


def _validate_schedule_boundary(boundary: JsonObject, label: str) -> None:
    """Validate one authored percent and converted sigma boundary."""

    start_percent = boundary.get("start_percent")
    start_sigma = boundary.get("start_sigma")
    strength = boundary.get("strength_multiplier")
    guarantee_steps = boundary.get("guarantee_steps")
    finite_numbers = (start_percent, start_sigma, strength)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        for value in finite_numbers
    ):
        raise ValueError(f"SDXL {label} schedule boundary is invalid.")
    assert isinstance(start_percent, int | float)
    assert isinstance(start_sigma, int | float)
    if not 0.0 <= float(start_percent) <= 1.0 or float(start_sigma) < 0.0:
        raise ValueError(f"SDXL {label} schedule boundary is invalid.")
    if (
        isinstance(guarantee_steps, bool)
        or not isinstance(guarantee_steps, int)
        or guarantee_steps < 0
    ):
        raise ValueError(f"SDXL {label} schedule guarantee is invalid.")


def sdxl_image_evidence(data: bytes) -> tuple[tuple[int, int], int]:
    """Return decoded dimensions and full RGB dynamic range."""

    with Image.open(BytesIO(data)) as image:
        rgb = image.convert("RGB")
        extrema = cast(
            tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
            rgb.getextrema(),
        )
        dynamic_range = max(high for _low, high in extrema) - min(
            low for low, _high in extrema
        )
        return rgb.size, dynamic_range
