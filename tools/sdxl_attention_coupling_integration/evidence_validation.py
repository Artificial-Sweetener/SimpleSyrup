# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate managed SDXL image, runtime, and regional diagnostic evidence."""

from __future__ import annotations

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
        raise ValueError(f"SDXL {label} diagnostics must contain snapshots.")
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
