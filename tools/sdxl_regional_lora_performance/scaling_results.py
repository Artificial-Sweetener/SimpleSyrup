# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Evaluate SDXL zero/one/four active-use performance gates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .scaling_cases import SdxlRegionalScalingProfile

ONE_USE_OVERHEAD_LIMIT_PERCENT = 15.0
FOUR_USE_OVERHEAD_LIMIT_PERCENT = 35.0


@dataclass(frozen=True, slots=True)
class SdxlActiveUseScalingSummary:
    """Retain scaling medians, overheads, and independent decisions."""

    zero_use_median_ms: float
    one_use_median_ms: float
    four_use_median_ms: float
    one_use_overhead_percent: float
    four_use_overhead_percent: float
    one_use_gate_passed: bool
    four_use_gate_passed: bool
    all_gates_passed: bool


def summarize_active_use_scaling(
    observations: Mapping[str, Mapping[str, object]],
) -> SdxlActiveUseScalingSummary:
    """Calculate exact one-use and four-use marginal overhead gates."""

    zero = _required_median(observations, SdxlRegionalScalingProfile.ZERO)
    one = _required_median(observations, SdxlRegionalScalingProfile.ONE)
    four = _required_median(observations, SdxlRegionalScalingProfile.FOUR)
    one_overhead = ((one / zero) - 1.0) * 100.0
    four_overhead = ((four / zero) - 1.0) * 100.0
    one_passed = _within_limit(one_overhead, ONE_USE_OVERHEAD_LIMIT_PERCENT)
    four_passed = _within_limit(four_overhead, FOUR_USE_OVERHEAD_LIMIT_PERCENT)
    return SdxlActiveUseScalingSummary(
        zero_use_median_ms=zero,
        one_use_median_ms=one,
        four_use_median_ms=four,
        one_use_overhead_percent=one_overhead,
        four_use_overhead_percent=four_overhead,
        one_use_gate_passed=one_passed,
        four_use_gate_passed=four_passed,
        all_gates_passed=one_passed and four_passed,
    )


def _required_median(
    observations: Mapping[str, Mapping[str, object]],
    profile: SdxlRegionalScalingProfile,
) -> float:
    """Return one positive median or reject an incomplete matrix."""

    observation = observations.get(profile.value)
    if observation is None:
        raise ValueError(
            f"Active-use scaling is missing required profile: {profile.value}."
        )
    value = observation.get("median_runtime_ms")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Scaling median for {profile.value} must be numeric.")
    median = float(value)
    if median <= 0.0:
        raise ValueError(f"Scaling median for {profile.value} must be positive.")
    return median


def _within_limit(value: float, limit: float) -> bool:
    """Apply one inclusive percentage gate without binary-boundary drift."""

    return value <= limit or math.isclose(
        value,
        limit,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
