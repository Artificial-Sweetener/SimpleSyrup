# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify model-neutral zero/one/four SDXL scaling declarations."""

from __future__ import annotations

from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_regional_lora_performance.scaling_cases import (
    SdxlRegionalScalingProfile,
    regional_scaling_cases,
)


def test_scaling_cases_change_only_exact_full_strength_adapter_count() -> None:
    """Keep prompt topology identical while declaring zero, one, and four uses."""

    cases = regional_scaling_cases(
        _prompts(),
        left_trigger_g="left trigger g",
        left_trigger_l="left trigger l",
        right_trigger_g="right trigger g",
        right_trigger_l="right trigger l",
        style_trigger_g="style trigger g",
        style_trigger_l="style trigger l",
    )

    assert tuple(item.profile for item in cases) == tuple(SdxlRegionalScalingProfile)
    assert tuple(item.adapter_count for item in cases) == (0, 1, 4)
    reference = cases[0].case
    for item in cases:
        case = item.case
        assert case.base_positive_g == reference.base_positive_g
        assert case.base_positive_l == reference.base_positive_l
        assert case.left_g == reference.left_g
        assert case.left_l == reference.left_l
        assert case.right_g == reference.right_g
        assert case.right_l == reference.right_l
        for adapter in (*case.left_adapters, *case.right_adapters):
            assert adapter.model_strength == 1.0
            assert adapter.clip_strength == 1.0
            assert adapter.schedule == ((0.0, 1.0),)


def _prompts() -> SdxlVisualPromptSet:
    """Return one generic dual-encoder prompt fixture."""

    return SdxlVisualPromptSet(
        base_positive_g="base g",
        base_positive_l="base l",
        base_negative_g="negative g",
        base_negative_l="negative l",
        left_positive_g="left g",
        left_positive_l="left l",
        right_positive_g="right g",
        right_positive_l="right l",
        left_negative_g="left negative g",
        left_negative_l="left negative l",
        right_negative_g="right negative g",
        right_negative_l="right negative l",
    )
