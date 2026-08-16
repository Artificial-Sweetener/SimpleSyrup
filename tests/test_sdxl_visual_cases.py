# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the fixed native-SDXL visual acceptance declarations."""

from __future__ import annotations

from pathlib import Path

from sdxl_visual_test_inventory import visual_inventory

from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    RIGHT_CHARACTER_SELECTION,
    STYLE_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    VisualMaskProfile,
    VisualMode,
)
from tools.sdxl_attention_coupling_integration.visual_cases import (
    visual_cases,
)
from tools.sdxl_attention_coupling_integration.visual_prompt_defaults import (
    BASE_NEGATIVE_G,
    BASE_NEGATIVE_L,
    BASE_POSITIVE_L,
    LEFT_BASE_L,
    RIGHT_BASE_L,
)


def test_visual_baseline_uses_explicit_multi_subject_tags(tmp_path: Path) -> None:
    """Keep the ownership matrix anchored to a visible two-subject composition."""

    assert "2girls" in BASE_POSITIVE_L
    assert "both women visible" in BASE_POSITIVE_L
    assert "1girl" in BASE_NEGATIVE_L
    assert "solo" in BASE_NEGATIVE_L
    assert "3girls" in BASE_NEGATIVE_L
    assert "crowd" in BASE_NEGATIVE_L
    assert BASE_NEGATIVE_G.startswith("bad quality, worst quality")
    cases = visual_cases(visual_inventory(tmp_path))
    by_id = {case.case_id: case for case in cases}
    assert "1girl" in LEFT_BASE_L
    assert "1girl" in RIGHT_BASE_L
    assert "1girl" in by_id["character-left"].left_l
    assert "1girl" in by_id["character-right"].right_l


def test_visual_matrix_has_twenty_eight_cases_and_thirty_outputs(
    tmp_path: Path,
) -> None:
    """Cover every required scenario without multiplying upscale refinements."""

    cases = visual_cases(visual_inventory(tmp_path))

    assert len(cases) == 28
    assert sum(len(case.modes) for case in cases) == 30
    assert len({case.case_id for case in cases}) == len(cases)
    assert cases[0].case_id == "baseline"
    assert cases[0].global_adapters == ()
    assert cases[0].left_adapters == ()
    assert cases[0].right_adapters == ()


def test_global_regional_and_schedule_contracts_remain_distinct(
    tmp_path: Path,
) -> None:
    """Preserve ordinary global use, authored regional reuse, and keyframes."""

    inventory = visual_inventory(tmp_path)
    by_id = {case.case_id: case for case in visual_cases(inventory)}
    prompt_control = by_id["global-style-right-character-prompt-control"]
    combined = by_id["global-style-regional-character"]
    assert prompt_control.left_l == combined.left_l
    assert prompt_control.left_g == combined.left_g
    assert prompt_control.right_l == combined.right_l
    assert prompt_control.right_g == combined.right_g
    assert prompt_control.global_style_l == combined.global_style_l
    assert prompt_control.global_style_g == combined.global_style_g
    assert prompt_control.global_adapters == combined.global_adapters
    assert prompt_control.left_adapters == combined.left_adapters == ()
    assert prompt_control.right_adapters == ()
    assert [adapter.lora_name for adapter in combined.global_adapters] == [
        STYLE_SELECTION
    ]
    assert [adapter.lora_name for adapter in combined.right_adapters] == [
        RIGHT_CHARACTER_SELECTION
    ]
    assert combined.modes == (VisualMode.FULL,)
    assert prompt_control.regional_prompt_weight == 1.0
    assert combined.regional_prompt_weight == 1.0
    control = by_id["global-style-control"]
    assert control.global_style_g == inventory.style.prompt_g
    assert control.global_style_l == inventory.style.prompt_l
    full_regional = by_id["global-style-full-regional-control"]
    assert full_regional.label != control.label
    assert full_regional.left_l == control.left_l
    assert full_regional.left_g == control.left_g
    assert full_regional.right_l == control.right_l
    assert full_regional.right_g == control.right_g
    assert full_regional.global_style_l == control.global_style_l
    assert full_regional.global_style_g == control.global_style_g
    assert full_regional.global_adapters == control.global_adapters
    assert full_regional.left_adapters == control.left_adapters
    assert full_regional.right_adapters == control.right_adapters
    assert full_regional.regional_prompt_start_percent == (
        control.regional_prompt_start_percent
    )
    assert full_regional.regional_prompt_weight == 1.0
    assert control.regional_prompt_weight == 0.4
    full_strength_case_ids = {
        "global-style-full-regional-control",
        "global-style-right-character-prompt-control",
        "global-style-regional-character",
        "same-style-global-left",
        "multiple-left-model-only-style",
        "multiple-left-scheduled-style-only",
        "multiple-left-midpoint-style-only",
        "multiple-left-minimal-window-style-only",
        "same-style-both",
    }
    assert all(
        case.regional_prompt_weight == 1.0
        for case in visual_cases(inventory)
        if case.case_id in full_strength_case_ids
    )
    assert all(
        case.regional_prompt_weight == 0.4
        for case in visual_cases(inventory)
        if case.case_id not in full_strength_case_ids
    )
    assert full_regional.mask_profile is control.mask_profile
    assert full_regional.region_mask_feather == control.region_mask_feather
    assert full_regional.modes == control.modes
    layout_first = by_id["global-style-layout-first-control"]
    assert layout_first.global_adapters == control.global_adapters
    assert layout_first.global_style_g == control.global_style_g
    assert layout_first.global_style_l == control.global_style_l
    assert layout_first.regional_prompt_start_percent == 0.15
    assert all(
        case.regional_prompt_start_percent == 0.0
        for case in visual_cases(inventory)
        if case.case_id != layout_first.case_id
    )
    spatial = by_id["spatial-mode-global-style-regional-character"]
    assert spatial.modes == (
        VisualMode.FULL,
        VisualMode.TILED,
        VisualMode.CONTEXTUAL,
    )
    same = by_id["same-style-both"]
    assert same.left_adapters[0].lora_name == same.right_adapters[0].lora_name
    scheduled = by_id["scheduled-right-character"].right_adapters[0]
    assert scheduled.schedule == ((0.0, 1.0), (0.6, 0.0))
    assert by_id["soft-overlap"].mask_profile is VisualMaskProfile.SOFT_OVERLAP
    assert by_id["soft-overlap"].region_mask_feather == 32
    assert by_id["uncovered-center"].mask_profile is VisualMaskProfile.UNCOVERED_CENTER
    assert by_id["baseline"].region_mask_feather == 0
    assert by_id["character-left"].region_mask_feather == 0
    assert by_id["character-right"].region_mask_feather == 0


def test_regional_style_has_symmetric_left_and_right_cases(tmp_path: Path) -> None:
    """Apply one generic style adapter independently to either named region."""

    inventory = visual_inventory(tmp_path)
    by_id = {case.case_id: case for case in visual_cases(inventory)}
    left = by_id["regional-style"]
    right = by_id["regional-style-right"]

    assert [adapter.lora_name for adapter in left.left_adapters] == [STYLE_SELECTION]
    assert left.right_adapters == ()
    assert right.left_adapters == ()
    assert [adapter.lora_name for adapter in right.right_adapters] == [STYLE_SELECTION]
    assert inventory.style.prompt_l in left.left_l
    assert inventory.style.prompt_l not in left.right_l
    assert inventory.style.prompt_l not in right.left_l
    assert inventory.style.prompt_l in right.right_l
    assert inventory.style.prompt_g not in left.base_positive_g
    assert inventory.style.prompt_g not in right.base_positive_g
    assert inventory.style.prompt_g in left.left_g
    assert inventory.style.prompt_g not in left.right_g
    assert inventory.style.prompt_g not in right.left_g
    assert inventory.style.prompt_g in right.right_g


def test_character_prompt_controls_change_only_the_adapter_axis(
    tmp_path: Path,
) -> None:
    """Pair each character LoRA case with its exact prompt-only control."""

    inventory = visual_inventory(tmp_path)
    by_id = {case.case_id: case for case in visual_cases(inventory)}
    for side in ("left", "right"):
        control = by_id[f"character-{side}-prompt-control"]
        regional = by_id[f"character-{side}"]
        assert control.left_l == regional.left_l
        assert control.left_g == regional.left_g
        assert control.right_l == regional.right_l
        assert control.right_g == regional.right_g
        assert control.global_adapters == regional.global_adapters == ()
        assert control.left_adapters == ()
        assert control.right_adapters == ()

    right_control = by_id["character-right-prompt-control"]
    right_regional = by_id["character-right"]
    assert right_control.left_l == inventory.left_character.prompt_l
    assert right_control.left_g == inventory.left_character.prompt_g
    assert right_regional.left_l == inventory.left_character.prompt_l
    assert right_regional.left_g == inventory.left_character.prompt_g
    assert right_regional.left_adapters == ()
    assert [adapter.lora_name for adapter in right_regional.right_adapters] == [
        RIGHT_CHARACTER_SELECTION
    ]

    control = by_id["different-characters-prompt-control"]
    regional = by_id["different-characters"]
    assert control.left_l == regional.left_l
    assert control.left_g == regional.left_g
    assert control.right_l == regional.right_l
    assert control.right_g == regional.right_g
    assert control.left_g == inventory.left_character.prompt_g
    assert control.right_g == inventory.right_character.prompt_g
    assert control.left_adapters == ()
    assert control.right_adapters == ()
