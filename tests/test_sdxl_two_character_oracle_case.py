# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lock the model-neutral SDXL two-character oracle declaration."""

from __future__ import annotations

from pathlib import Path

from sdxl_visual_test_inventory import visual_inventory

from tools.sdxl_attention_coupling_integration.sampling_controls import (
    SDXL_VISUAL_SAMPLING,
)
from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    LEFT_CHARACTER_SELECTION,
    RIGHT_CHARACTER_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    VisualMaskProfile,
    VisualMode,
)
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_two_character_oracle import (
    SDXL_TWO_CHARACTER_ORACLE_ID,
    two_character_oracle_cases,
)


def test_two_character_oracle_locks_accepted_graph_without_model_identity(
    tmp_path: Path,
) -> None:
    """Require the accepted full-strength topology and sampling controls."""

    inventory = visual_inventory(tmp_path)
    prompt_values = tuple(f"authored-prompt-{index}" for index in range(12))
    prompts = SdxlVisualPromptSet(*prompt_values)

    (case,) = two_character_oracle_cases(inventory, prompts)

    assert case.case_id == SDXL_TWO_CHARACTER_ORACLE_ID
    assert case.global_adapters == ()
    assert case.mask_profile is VisualMaskProfile.HARD
    assert case.region_mask_feather == 0
    assert case.modes == (VisualMode.FULL,)
    assert case.regional_prompt_start_percent == 0.0
    assert case.regional_prompt_weight == 1.0
    assert tuple(item.lora_name for item in case.left_adapters) == (
        LEFT_CHARACTER_SELECTION,
    )
    assert tuple(item.lora_name for item in case.right_adapters) == (
        RIGHT_CHARACTER_SELECTION,
    )
    for adapter in (*case.left_adapters, *case.right_adapters):
        assert adapter.model_strength == 1.0
        assert adapter.clip_strength == 1.0
        assert adapter.schedule == ((0.0, 1.0),)
    assert (
        case.base_positive_g,
        case.base_positive_l,
        case.base_negative_g,
        case.base_negative_l,
        case.left_negative_g,
        case.left_negative_l,
        case.right_negative_g,
        case.right_negative_l,
    ) == (
        prompts.base_positive_g,
        prompts.base_positive_l,
        prompts.base_negative_g,
        prompts.base_negative_l,
        prompts.left_negative_g,
        prompts.left_negative_l,
        prompts.right_negative_g,
        prompts.right_negative_l,
    )
    assert case.left_g == inventory.left_character.prompt_g
    assert case.left_l == inventory.left_character.prompt_l
    assert case.right_g == inventory.right_character.prompt_g
    assert case.right_l == inventory.right_character.prompt_l
    assert SDXL_VISUAL_SAMPLING.seed == 7_429_113_057
    assert SDXL_VISUAL_SAMPLING.steps == 30
    assert SDXL_VISUAL_SAMPLING.cfg == 5.0
    assert SDXL_VISUAL_SAMPLING.sampler == "euler_ancestral"
    assert SDXL_VISUAL_SAMPLING.scheduler == "karras"
