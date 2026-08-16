# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derive RA-07 decisions from four matched warmed timing modes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .conventional_variant_workflow import ConventionalVariantBranch
from .two_adapter_workflow import TwoAdapterPerformanceMode

RESIDUAL_PROFILE_THRESHOLD_PERCENT = 5.0
PERFORMANCE_TERMINAL_RATIO = 2.93


@dataclass(frozen=True, slots=True)
class SdxlMatchedFloorSummary:
    """Retain derived performance-floor metrics and gate decisions."""

    two_variant_floor_ms: float
    regional_to_vanilla_ratio: float
    regional_excess_above_floor_ms: float
    regional_excess_percent: float
    residual_profile_required: bool
    performance_terminal_passed: bool


def summarize_matched_floor(
    observations: Mapping[str, Mapping[str, object]],
) -> SdxlMatchedFloorSummary:
    """Calculate the exact floor, residual, and terminal decisions."""

    global_median = _required_median(
        observations,
        TwoAdapterPerformanceMode.GLOBAL_REFERENCE.value,
    )
    left_median = _required_median(
        observations,
        ConventionalVariantBranch.LEFT.value,
    )
    right_median = _required_median(
        observations,
        ConventionalVariantBranch.RIGHT.value,
    )
    regional_median = _required_median(
        observations,
        TwoAdapterPerformanceMode.REGIONAL.value,
    )
    floor = left_median + right_median
    excess = regional_median - floor
    excess_percent = excess / regional_median * 100.0
    regional_ratio = regional_median / global_median
    return SdxlMatchedFloorSummary(
        two_variant_floor_ms=floor,
        regional_to_vanilla_ratio=regional_ratio,
        regional_excess_above_floor_ms=excess,
        regional_excess_percent=excess_percent,
        residual_profile_required=(
            excess_percent >= RESIDUAL_PROFILE_THRESHOLD_PERCENT
        ),
        performance_terminal_passed=regional_ratio <= PERFORMANCE_TERMINAL_RATIO,
    )


def _required_median(
    observations: Mapping[str, Mapping[str, object]],
    mode: str,
) -> float:
    """Return one positive finite median or reject the partial matrix."""

    observation = observations.get(mode)
    if observation is None:
        raise ValueError(f"Matched floor is missing required mode: {mode}.")
    value = observation.get("median_runtime_ms")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Matched floor median for {mode} must be numeric.")
    median = float(value)
    if median <= 0.0:
        raise ValueError(f"Matched floor median for {mode} must be positive.")
    return median
