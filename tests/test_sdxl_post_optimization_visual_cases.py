# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lock the two decisive post-optimization SDXL visual cases."""

from __future__ import annotations

from dataclasses import replace
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
from tools.sdxl_post_optimization_visual_proof.cases import (
    post_optimization_visual_cases,
)
from tools.sdxl_two_character_oracle import two_character_oracle_cases


def test_post_optimization_cases_lock_placement_strength_and_style_prefix(
    tmp_path: Path,
) -> None:
    """Require two full-strength cases with the style trigger first globally."""

    inventory = visual_inventory(tmp_path)
    prompts = SdxlVisualPromptSet(*tuple(f"prompt-{index}" for index in range(12)))
    simultaneous, global_style_right = post_optimization_visual_cases(
        inventory,
        prompts,
    )
    (oracle,) = two_character_oracle_cases(inventory, prompts)

    assert simultaneous.case_id == "post-cache-simultaneous-characters"
    assert simultaneous == replace(
        oracle,
        case_id="post-cache-simultaneous-characters",
        label=(
            f"POST-CACHE — {inventory.left_character.label} left 1.0; "
            f"{inventory.right_character.label} right 1.0"
        ),
    )
    assert tuple(item.lora_name for item in simultaneous.left_adapters) == (
        LEFT_CHARACTER_SELECTION,
    )
    assert tuple(item.lora_name for item in simultaneous.right_adapters) == (
        RIGHT_CHARACTER_SELECTION,
    )
    assert global_style_right.case_id == "post-cache-global-style-right-character"
    assert global_style_right.left_adapters == ()
    assert tuple(item.lora_name for item in global_style_right.right_adapters) == (
        RIGHT_CHARACTER_SELECTION,
    )
    assert global_style_right.global_adapters == (
        type(global_style_right.global_adapters[0])(STYLE_SELECTION, 1.0),
    )
    assert global_style_right.base_positive_g.startswith(inventory.style.prompt_g)
    assert global_style_right.base_positive_l.startswith(inventory.style.prompt_l)
    for case in (simultaneous, global_style_right):
        for adapter in (*case.left_adapters, *case.right_adapters):
            assert adapter.model_strength == 1.0
            assert adapter.clip_strength == 1.0
        assert case.regional_prompt_weight == 1.0
