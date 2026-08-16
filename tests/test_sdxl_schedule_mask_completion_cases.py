# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lock the model-neutral RA-06 visual proof declarations."""

from __future__ import annotations

from pathlib import Path

from sdxl_visual_test_inventory import visual_inventory

from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    LEFT_CHARACTER_SELECTION,
    RIGHT_CHARACTER_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    VisualMaskProfile,
)
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_schedule_mask_completion.cases import (
    EARLY_DOMINANT_SCHEDULE,
    FULL_STRENGTH_SCHEDULE,
    LATE_DOMINANT_SCHEDULE,
    schedule_mask_completion_cases,
)


def test_cases_isolate_independent_schedules_and_mask_geometry(tmp_path: Path) -> None:
    """Keep one sampling axis different in each focused RA-06 case."""

    inventory = visual_inventory(tmp_path)
    cases = schedule_mask_completion_cases(inventory, _prompts())
    by_id = {case.case_id: case for case in cases}

    assert tuple(by_id) == (
        "ra06-independent-schedules",
        "ra06-soft-overlap",
        "ra06-uncovered-center",
    )
    scheduled = by_id["ra06-independent-schedules"]
    assert scheduled.left_adapters[0].schedule == EARLY_DOMINANT_SCHEDULE
    assert scheduled.right_adapters[0].schedule == LATE_DOMINANT_SCHEDULE
    assert scheduled.mask_profile is VisualMaskProfile.HARD

    soft = by_id["ra06-soft-overlap"]
    uncovered = by_id["ra06-uncovered-center"]
    assert soft.mask_profile is VisualMaskProfile.SOFT_OVERLAP
    assert soft.region_mask_feather == 32
    assert uncovered.mask_profile is VisualMaskProfile.UNCOVERED_CENTER
    for case in (soft, uncovered):
        assert case.left_adapters[0].schedule == FULL_STRENGTH_SCHEDULE
        assert case.right_adapters[0].schedule == FULL_STRENGTH_SCHEDULE
        assert case.left_adapters[0].lora_name == LEFT_CHARACTER_SELECTION
        assert case.right_adapters[0].lora_name == RIGHT_CHARACTER_SELECTION
        assert case.left_adapters[0].model_strength == 1.0
        assert case.left_adapters[0].clip_strength == 1.0
        assert case.right_adapters[0].model_strength == 1.0
        assert case.right_adapters[0].clip_strength == 1.0
    assert all(case.global_adapters == () for case in cases)
    assert all(case.regional_prompt_weight == 1.0 for case in cases)
    assert all(case.left_g == inventory.left_character.prompt_g for case in cases)
    assert all(case.right_g == inventory.right_character.prompt_g for case in cases)


def _prompts() -> SdxlVisualPromptSet:
    """Return one anonymous complete two-region prompt fixture."""

    return SdxlVisualPromptSet(
        base_positive_g="global positive g",
        base_positive_l="global positive l",
        base_negative_g="global negative g",
        base_negative_l="global negative l",
        left_positive_g="left positive g",
        left_positive_l="left positive l",
        right_positive_g="right positive g",
        right_positive_l="right positive l",
        left_negative_g="left negative g",
        left_negative_l="left negative l",
        right_negative_g="right negative g",
        right_negative_l="right negative l",
    )
