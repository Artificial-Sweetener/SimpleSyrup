# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify SDXL zero/one/four active-use performance decisions."""

from __future__ import annotations

import pytest

from tools.sdxl_regional_lora_performance.scaling_cases import (
    SdxlRegionalScalingProfile,
)
from tools.sdxl_regional_lora_performance.scaling_results import (
    summarize_active_use_scaling,
)


def test_scaling_summary_applies_exact_one_and_four_use_gates() -> None:
    """Pass one use at 15% and four uses at 35% inclusive."""

    summary = summarize_active_use_scaling(
        {
            SdxlRegionalScalingProfile.ZERO.value: _observation(10_000.0),
            SdxlRegionalScalingProfile.ONE.value: _observation(11_500.0),
            SdxlRegionalScalingProfile.FOUR.value: _observation(13_500.0),
        }
    )

    assert summary.one_use_overhead_percent == pytest.approx(15.0)
    assert summary.four_use_overhead_percent == pytest.approx(35.0)
    assert summary.one_use_gate_passed is True
    assert summary.four_use_gate_passed is True
    assert summary.all_gates_passed is True


def test_scaling_summary_fails_only_the_exceeded_gate() -> None:
    """Retain independent decisions for one-use and four-use overhead."""

    summary = summarize_active_use_scaling(
        {
            SdxlRegionalScalingProfile.ZERO.value: _observation(10_000.0),
            SdxlRegionalScalingProfile.ONE.value: _observation(11_600.0),
            SdxlRegionalScalingProfile.FOUR.value: _observation(13_400.0),
        }
    )

    assert summary.one_use_gate_passed is False
    assert summary.four_use_gate_passed is True
    assert summary.all_gates_passed is False


def test_scaling_summary_rejects_partial_profile_matrix() -> None:
    """Fail closed before publishing an incomplete scaling decision."""

    with pytest.raises(ValueError, match="missing required profile"):
        summarize_active_use_scaling(
            {
                SdxlRegionalScalingProfile.ZERO.value: _observation(10_000.0),
            }
        )


def _observation(median_runtime_ms: float) -> dict[str, object]:
    """Build one generic recorded median."""

    return {"median_runtime_ms": median_runtime_ms}
