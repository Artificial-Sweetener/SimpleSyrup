# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declare current-route SDXL schedule and mask-geometry proof cases."""

from __future__ import annotations

from dataclasses import replace

from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    LEFT_CHARACTER_SELECTION,
    RIGHT_CHARACTER_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    RegionalVisualAdapter,
    SdxlVisualCase,
    VisualMaskProfile,
)
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)

EARLY_DOMINANT_SCHEDULE = ((0.0, 1.0), (0.5, 0.5))
LATE_DOMINANT_SCHEDULE = ((0.0, 0.5), (0.5, 1.0))
FULL_STRENGTH_SCHEDULE = ((0.0, 1.0),)


def schedule_mask_completion_cases(
    inventory: SdxlVisualInventory,
    prompts: SdxlVisualPromptSet,
) -> tuple[SdxlVisualCase, ...]:
    """Return the independent-schedule and geometry completion cases."""

    base = replace(
        prompts.control_case(
            "ra06-character-control",
            "RA-06 full-strength character control",
            regional_prompt_weight=1.0,
        ),
        left_g=inventory.left_character.prompt_g,
        left_l=inventory.left_character.prompt_l,
        right_g=inventory.right_character.prompt_g,
        right_l=inventory.right_character.prompt_l,
    )
    constant_left = RegionalVisualAdapter(
        LEFT_CHARACTER_SELECTION,
        1.0,
        1.0,
        FULL_STRENGTH_SCHEDULE,
    )
    constant_right = RegionalVisualAdapter(
        RIGHT_CHARACTER_SELECTION,
        1.0,
        1.0,
        FULL_STRENGTH_SCHEDULE,
    )
    constant_pair = (constant_left, constant_right)
    return (
        replace(
            base,
            case_id="ra06-independent-schedules",
            label=(
                "Independent schedules: left 1.0 to 0.5; right 0.5 to 1.0 at 50 percent"
            ),
            left_adapters=(replace(constant_left, schedule=EARLY_DOMINANT_SCHEDULE),),
            right_adapters=(replace(constant_right, schedule=LATE_DOMINANT_SCHEDULE),),
        ),
        replace(
            base,
            case_id="ra06-soft-overlap",
            label="Full-strength characters with feathered overlapping masks",
            left_adapters=(constant_pair[0],),
            right_adapters=(constant_pair[1],),
            mask_profile=VisualMaskProfile.SOFT_OVERLAP,
            region_mask_feather=32,
        ),
        replace(
            base,
            case_id="ra06-uncovered-center",
            label="Full-strength characters with globally filled uncovered center",
            left_adapters=(constant_pair[0],),
            right_adapters=(constant_pair[1],),
            mask_profile=VisualMaskProfile.UNCOVERED_CENTER,
        ),
    )
