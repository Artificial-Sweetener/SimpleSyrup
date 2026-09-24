# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the RA-07 matched warmed-floor decision math."""

from __future__ import annotations

import pytest

from tools.sdxl_regional_lora_performance.conventional_variant_workflow import (
    ConventionalVariantBranch,
)
from tools.sdxl_regional_lora_performance.matched_floor_results import (
    summarize_matched_floor,
)
from tools.sdxl_regional_lora_performance.two_adapter_workflow import (
    TwoAdapterPerformanceMode,
)


def test_summary_reports_global_ratio_and_residual_above_two_variant_floor() -> None:
    """Derive the exact >=5-percent decision from four matched medians."""

    summary = summarize_matched_floor(
        {
            TwoAdapterPerformanceMode.GLOBAL_REFERENCE.value: _observation(6_000.0),
            ConventionalVariantBranch.LEFT.value: _observation(4_200.0),
            ConventionalVariantBranch.RIGHT.value: _observation(4_300.0),
            TwoAdapterPerformanceMode.REGIONAL.value: _observation(9_000.0),
        }
    )

    assert summary.two_variant_floor_ms == 8_500.0
    assert summary.regional_to_vanilla_ratio == 1.5
    assert summary.regional_excess_above_floor_ms == 500.0
    assert summary.regional_excess_percent == pytest.approx(5.5555555556)
    assert summary.residual_profile_required is True
    assert summary.performance_terminal_passed is True


def test_summary_closes_sub_five_percent_residual_without_hiding_negative_delta() -> (
    None
):
    """Treat a faster composed path as measured evidence, not zeroed arithmetic."""

    summary = summarize_matched_floor(
        {
            TwoAdapterPerformanceMode.GLOBAL_REFERENCE.value: _observation(4_000.0),
            ConventionalVariantBranch.LEFT.value: _observation(4_500.0),
            ConventionalVariantBranch.RIGHT.value: _observation(4_500.0),
            TwoAdapterPerformanceMode.REGIONAL.value: _observation(8_800.0),
        }
    )

    assert summary.regional_excess_above_floor_ms == -200.0
    assert summary.regional_excess_percent == pytest.approx(-2.2727272727)
    assert summary.residual_profile_required is False
    assert summary.performance_terminal_passed is True


def test_summary_fails_closed_when_a_required_mode_is_missing() -> None:
    """Reject partial matrices before drawing a performance conclusion."""

    with pytest.raises(ValueError, match="missing required mode"):
        summarize_matched_floor(
            {TwoAdapterPerformanceMode.GLOBAL_REFERENCE.value: _observation(4_000.0)}
        )


def _observation(median_runtime_ms: float) -> dict[str, object]:
    """Build one generic recorded-median payload."""

    return {"median_runtime_ms": median_runtime_ms}
