# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declare the bounded two-region full-strength character-LoRA matrix."""

from __future__ import annotations

from dataclasses import replace

from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    LEFT_CHARACTER_SELECTION,
    RIGHT_CHARACTER_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    RegionalVisualAdapter,
    SdxlVisualCase,
)
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)

FULL_STRENGTH_SCHEDULE = ((0.0, 1.0),)


def full_strength_composition_cases(
    inventory: SdxlVisualInventory,
    prompts: SdxlVisualPromptSet,
) -> tuple[SdxlVisualCase, ...]:
    """Return left-only, right-only, and simultaneous full-strength cases."""

    control = SdxlVisualCase(
        "full-strength-control",
        "Full-strength composition control",
        base_positive_g=prompts.base_positive_g,
        base_positive_l=prompts.base_positive_l,
        base_negative_g=prompts.base_negative_g,
        base_negative_l=prompts.base_negative_l,
        left_g=prompts.left_positive_g,
        left_l=prompts.left_positive_l,
        right_g=prompts.right_positive_g,
        right_l=prompts.right_positive_l,
        left_negative_g=prompts.left_negative_g,
        left_negative_l=prompts.left_negative_l,
        right_negative_g=prompts.right_negative_g,
        right_negative_l=prompts.right_negative_l,
        regional_prompt_weight=1.0,
    )
    left = RegionalVisualAdapter(
        LEFT_CHARACTER_SELECTION,
        1.0,
        1.0,
        FULL_STRENGTH_SCHEDULE,
    )
    right = RegionalVisualAdapter(
        RIGHT_CHARACTER_SELECTION,
        1.0,
        1.0,
        FULL_STRENGTH_SCHEDULE,
    )
    return (
        replace(
            control,
            case_id="full-strength-left-only",
            label=(
                f"{inventory.left_character.label} left at 1.0; "
                "pink/black right control"
            ),
            left_g=inventory.left_character.prompt_g,
            left_l=inventory.left_character.prompt_l,
            left_adapters=(left,),
        ),
        replace(
            control,
            case_id="full-strength-right-only",
            label=(
                "Pink/black left control; "
                f"{inventory.right_character.label} right at 1.0"
            ),
            right_g=inventory.right_character.prompt_g,
            right_l=inventory.right_character.prompt_l,
            right_adapters=(right,),
        ),
        replace(
            control,
            case_id="full-strength-simultaneous",
            label=(
                f"{inventory.left_character.label} left at 1.0; "
                f"{inventory.right_character.label} right at 1.0"
            ),
            left_g=inventory.left_character.prompt_g,
            left_l=inventory.left_character.prompt_l,
            right_g=inventory.right_character.prompt_g,
            right_l=inventory.right_character.prompt_l,
            left_adapters=(left,),
            right_adapters=(right,),
        ),
    )
