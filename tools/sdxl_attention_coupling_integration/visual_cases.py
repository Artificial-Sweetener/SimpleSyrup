# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declare the immutable U11 native-SDXL visual acceptance cases."""

from __future__ import annotations

from .visual_adapter_controls import (
    left_character_adapter,
    right_character_adapter,
)
from .visual_adapter_selections import (
    RIGHT_CHARACTER_SELECTION,
    STYLE_SELECTION,
)
from .visual_case_model import (
    GlobalVisualAdapter,
    RegionalVisualAdapter,
    SdxlVisualCase,
    VisualMaskProfile,
    VisualMode,
)
from .visual_distinct_character_cases import distinct_character_cases
from .visual_inventory import SdxlVisualInventory
from .visual_multiple_left_cases import multiple_left_cases
from .visual_prompt_defaults import (
    LEFT_BASE_G,
    LEFT_BASE_L,
    RIGHT_BASE_G,
    RIGHT_BASE_L,
)


def visual_cases(inventory: SdxlVisualInventory) -> tuple[SdxlVisualCase, ...]:
    """Return the generic matrix instantiated from one external inventory."""

    left_character = left_character_adapter()
    right_character = right_character_adapter()
    style = RegionalVisualAdapter(STYLE_SELECTION, 0.75, 0.75)
    style_left_l = f"{LEFT_BASE_L}, {inventory.style.prompt_l}"
    style_left_g = f"{LEFT_BASE_G}, {inventory.style.prompt_g}"
    style_right_l = f"{RIGHT_BASE_L}, {inventory.style.prompt_l}"
    style_right_g = f"{RIGHT_BASE_G}, {inventory.style.prompt_g}"
    return (
        SdxlVisualCase("baseline", "No LoRA baseline"),
        SdxlVisualCase(
            "character-left-prompt-control",
            f"{inventory.left_character.label} prompt on left without LoRA",
            left_l=inventory.left_character.prompt_l,
            left_g=inventory.left_character.prompt_g,
        ),
        SdxlVisualCase(
            "character-left",
            f"{inventory.left_character.label} on left only",
            left_l=inventory.left_character.prompt_l,
            left_g=inventory.left_character.prompt_g,
            left_adapters=(left_character,),
        ),
        SdxlVisualCase(
            "character-right-prompt-control",
            f"{inventory.right_character.label} prompt on right without LoRA",
            left_l=inventory.left_character.prompt_l,
            left_g=inventory.left_character.prompt_g,
            right_l=inventory.right_character.prompt_l,
            right_g=inventory.right_character.prompt_g,
        ),
        SdxlVisualCase(
            "character-right",
            f"{inventory.right_character.label} on right only",
            left_l=inventory.left_character.prompt_l,
            left_g=inventory.left_character.prompt_g,
            right_l=inventory.right_character.prompt_l,
            right_g=inventory.right_character.prompt_g,
            right_adapters=(right_character,),
        ),
        *distinct_character_cases(
            inventory,
            left_adapter=left_character,
            right_adapter=right_character,
        ),
        SdxlVisualCase(
            "global-style-control",
            f"Global {inventory.style.label} on both subjects",
            global_style_l=inventory.style.prompt_l,
            global_style_g=inventory.style.prompt_g,
            global_adapters=(GlobalVisualAdapter(STYLE_SELECTION, 0.65),),
        ),
        SdxlVisualCase(
            "global-style-full-regional-control",
            f"Global {inventory.style.label} with full regional prompt strength",
            global_style_l=inventory.style.prompt_l,
            global_style_g=inventory.style.prompt_g,
            global_adapters=(GlobalVisualAdapter(STYLE_SELECTION, 0.65),),
            regional_prompt_weight=1.0,
        ),
        SdxlVisualCase(
            "global-style-layout-first-control",
            f"Global {inventory.style.label} with regional prompts after 15%",
            global_style_l=inventory.style.prompt_l,
            global_style_g=inventory.style.prompt_g,
            global_adapters=(GlobalVisualAdapter(STYLE_SELECTION, 0.65),),
            regional_prompt_start_percent=0.15,
        ),
        SdxlVisualCase(
            "global-style-right-character-prompt-control",
            f"Global {inventory.style.label} with right character prompt",
            right_l=inventory.right_character.prompt_l,
            right_g=inventory.right_character.prompt_g,
            global_style_l=inventory.style.prompt_l,
            global_style_g=inventory.style.prompt_g,
            global_adapters=(GlobalVisualAdapter(STYLE_SELECTION, 0.65),),
            regional_prompt_weight=1.0,
        ),
        SdxlVisualCase(
            "global-style-regional-character",
            f"Global {inventory.style.label} with right character adapter",
            right_l=inventory.right_character.prompt_l,
            right_g=inventory.right_character.prompt_g,
            global_style_l=inventory.style.prompt_l,
            global_style_g=inventory.style.prompt_g,
            global_adapters=(GlobalVisualAdapter(STYLE_SELECTION, 0.65),),
            right_adapters=(right_character,),
            regional_prompt_weight=1.0,
        ),
        SdxlVisualCase(
            "spatial-mode-global-style-regional-character",
            f"Global {inventory.style.label} with right character adapter",
            right_l=inventory.right_character.prompt_l,
            right_g=inventory.right_character.prompt_g,
            global_style_l=inventory.style.prompt_l,
            global_style_g=inventory.style.prompt_g,
            global_adapters=(GlobalVisualAdapter(STYLE_SELECTION, 0.65),),
            right_adapters=(right_character,),
            modes=(VisualMode.FULL, VisualMode.TILED, VisualMode.CONTEXTUAL),
        ),
        SdxlVisualCase(
            "regional-style",
            f"{inventory.style.label} on left only",
            left_l=style_left_l,
            left_g=style_left_g,
            left_adapters=(style,),
        ),
        SdxlVisualCase(
            "regional-style-right",
            f"{inventory.style.label} on right only",
            right_l=style_right_l,
            right_g=style_right_g,
            right_adapters=(style,),
        ),
        SdxlVisualCase(
            "same-style-global-left",
            "Same style globally and stronger on left",
            left_l=style_left_l,
            left_g=style_left_g,
            global_style_l=inventory.style.prompt_l,
            global_style_g=inventory.style.prompt_g,
            global_adapters=(GlobalVisualAdapter(STYLE_SELECTION, 0.35),),
            left_adapters=(RegionalVisualAdapter(STYLE_SELECTION, 0.55, 0.55),),
            regional_prompt_weight=1.0,
        ),
        *multiple_left_cases(
            inventory,
            left_character=left_character,
        ),
        SdxlVisualCase(
            "same-style-both",
            "Same style adapter authored in both regions",
            left_l=style_left_l,
            left_g=style_left_g,
            right_l=style_right_l,
            right_g=style_right_g,
            left_adapters=(style,),
            right_adapters=(style,),
            regional_prompt_weight=1.0,
        ),
        SdxlVisualCase(
            "scheduled-right-character",
            "Right character active through 60 percent of denoising",
            right_l=inventory.right_character.prompt_l,
            right_g=inventory.right_character.prompt_g,
            right_adapters=(
                RegionalVisualAdapter(
                    RIGHT_CHARACTER_SELECTION,
                    0.9,
                    0.9,
                    ((0.0, 1.0), (0.6, 0.0)),
                ),
            ),
        ),
        SdxlVisualCase(
            "soft-overlap",
            "Different character adapters with soft overlap",
            left_l=inventory.left_character.prompt_l,
            left_g=inventory.left_character.prompt_g,
            right_l=inventory.right_character.prompt_l,
            right_g=inventory.right_character.prompt_g,
            left_adapters=(left_character,),
            right_adapters=(right_character,),
            mask_profile=VisualMaskProfile.SOFT_OVERLAP,
            region_mask_feather=32,
        ),
        SdxlVisualCase(
            "uncovered-center",
            "Different character adapters with uncovered center",
            left_l=inventory.left_character.prompt_l,
            left_g=inventory.left_character.prompt_g,
            right_l=inventory.right_character.prompt_l,
            right_g=inventory.right_character.prompt_g,
            left_adapters=(left_character,),
            right_adapters=(right_character,),
            mask_profile=VisualMaskProfile.UNCOVERED_CENTER,
        ),
    )
