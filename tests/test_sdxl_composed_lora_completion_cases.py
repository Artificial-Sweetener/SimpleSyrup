# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lock the unresolved current-route composed-LoRA proof cases."""

from __future__ import annotations

from pathlib import Path

from sdxl_visual_test_inventory import visual_inventory

from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    LEFT_CHARACTER_SELECTION,
    RIGHT_CHARACTER_SELECTION,
    STYLE_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_composed_lora_completion.cases import (
    REPEATED_GLOBAL_STRENGTH,
    REPEATED_REGIONAL_STRENGTH,
    composed_lora_completion_cases,
)


def test_completion_cases_lock_causal_order_scope_and_strength(tmp_path: Path) -> None:
    """Require exact adapter placement for all three unresolved gaps."""

    inventory = visual_inventory(tmp_path)
    prompts = _prompts()
    cases = composed_lora_completion_cases(inventory, prompts)
    by_id = {case.case_id: case for case in cases}

    assert tuple(by_id) == (
        "composed-multiple-left",
        "composed-global-style-control",
        "composed-same-style-global-control",
        "composed-same-style-global-left",
        "composed-same-style-regional-control",
        "composed-same-style-both-regions",
    )
    multiple = by_id["composed-multiple-left"]
    assert tuple(item.lora_name for item in multiple.left_adapters) == (
        LEFT_CHARACTER_SELECTION,
        STYLE_SELECTION,
    )
    assert tuple(item.lora_name for item in multiple.right_adapters) == (
        RIGHT_CHARACTER_SELECTION,
    )
    assert multiple.right_g == inventory.right_character.prompt_g
    assert multiple.right_l == inventory.right_character.prompt_l
    assert all(
        item.model_strength == item.clip_strength == 1.0
        for item in (*multiple.left_adapters, *multiple.right_adapters)
    )

    full_global = by_id["composed-global-style-control"]
    global_control = by_id["composed-same-style-global-control"]
    global_left = by_id["composed-same-style-global-left"]
    assert full_global.global_adapters[0].strength == 1.0
    assert global_control.base_positive_g == global_left.base_positive_g
    assert global_control.base_positive_l == global_left.base_positive_l
    assert global_control.global_adapters == global_left.global_adapters
    assert global_control.global_adapters[0].strength == REPEATED_GLOBAL_STRENGTH
    assert global_control.left_adapters == ()
    assert tuple(item.lora_name for item in global_left.left_adapters) == (
        STYLE_SELECTION,
    )
    assert global_left.left_adapters[0].model_strength == (REPEATED_REGIONAL_STRENGTH)
    assert global_left.left_adapters[0].clip_strength == (REPEATED_REGIONAL_STRENGTH)
    assert (
        global_control.global_adapters[0].strength
        + global_left.left_adapters[0].model_strength
        == 1.0
    )
    assert global_left.base_positive_g.count(inventory.style.prompt_g) == 1
    assert global_left.base_positive_l.count(inventory.style.prompt_l) == 1
    assert inventory.style.prompt_g not in global_left.left_g
    assert inventory.style.prompt_l not in global_left.left_l

    bilateral_control = by_id["composed-same-style-regional-control"]
    both = by_id["composed-same-style-both-regions"]
    assert bilateral_control.global_adapters == ()
    assert bilateral_control.left_adapters == ()
    assert bilateral_control.right_adapters == ()
    assert bilateral_control.base_positive_g == both.base_positive_g
    assert bilateral_control.base_positive_l == both.base_positive_l
    assert both.global_adapters == ()
    assert tuple(item.lora_name for item in both.left_adapters) == (STYLE_SELECTION,)
    assert tuple(item.lora_name for item in both.right_adapters) == (STYLE_SELECTION,)
    assert both.left_g.startswith(inventory.style.prompt_g)
    assert both.right_g.startswith(inventory.style.prompt_g)
    assert all(case.regional_prompt_weight == 1.0 for case in cases)


def test_prompt_set_materializes_an_exact_control_case() -> None:
    """Keep all global and section prompts in the shared control owner."""

    prompts = _prompts()
    case = prompts.control_case("anonymous-control", "Anonymous control")

    observed = (
        case.base_positive_g,
        case.base_positive_l,
        case.base_negative_g,
        case.base_negative_l,
        case.left_g,
        case.left_l,
        case.right_g,
        case.right_l,
        case.left_negative_g,
        case.left_negative_l,
        case.right_negative_g,
        case.right_negative_l,
    )
    expected = (
        prompts.base_positive_g,
        prompts.base_positive_l,
        prompts.base_negative_g,
        prompts.base_negative_l,
        prompts.left_positive_g,
        prompts.left_positive_l,
        prompts.right_positive_g,
        prompts.right_positive_l,
        prompts.left_negative_g,
        prompts.left_negative_l,
        prompts.right_negative_g,
        prompts.right_negative_l,
    )
    assert observed == expected


def _prompts() -> SdxlVisualPromptSet:
    """Return one anonymous complete SEP prompt fixture."""

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
