# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure geometry helpers for scale-factor detailing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetailScalePlan:
    """Describe the scaled crop dimensions used for one detail pass."""

    width: int
    height: int
    scale: float
    unclamped_long_side: float
    target_long_side: float


def build_detail_scale_plan(
    detected_width: int,
    detected_height: int,
    crop_width: int,
    crop_height: int,
    scale_factor: float,
    clamp_size: int,
) -> DetailScalePlan:
    """Calculate a scaled crop size from crop-region geometry."""

    _validate_positive_int("detected_width", detected_width)
    _validate_positive_int("detected_height", detected_height)
    _validate_positive_int("crop_width", crop_width)
    _validate_positive_int("crop_height", crop_height)
    if scale_factor <= 0.0:
        raise ValueError("scale_factor must be greater than 0.")
    if clamp_size < 0:
        raise ValueError("clamp_size must be 0 or greater. Use 0 for no clamp.")

    crop_long_side = float(max(crop_width, crop_height))
    unclamped_long_side = crop_long_side * float(scale_factor)
    target_long_side = unclamped_long_side
    if clamp_size > 0:
        target_long_side = min(unclamped_long_side, float(clamp_size))

    scale = target_long_side / crop_long_side
    width = max(1, int(round(crop_width * scale)))
    height = max(1, int(round(crop_height * scale)))
    return DetailScalePlan(
        width=width,
        height=height,
        scale=scale,
        unclamped_long_side=unclamped_long_side,
        target_long_side=target_long_side,
    )


def _validate_positive_int(name: str, value: int) -> None:
    """Reject non-positive integer dimensions."""

    if int(value) <= 0:
        raise ValueError(f"{name} must be greater than 0.")
