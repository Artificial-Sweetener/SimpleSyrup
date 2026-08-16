# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declare the unresolved current-route SDXL composed-LoRA cases."""

from __future__ import annotations

from dataclasses import replace

from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    LEFT_CHARACTER_SELECTION,
    RIGHT_CHARACTER_SELECTION,
    STYLE_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    GlobalVisualAdapter,
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
REPEATED_GLOBAL_STRENGTH = 0.45
REPEATED_REGIONAL_STRENGTH = 0.55


def composed_lora_completion_cases(
    inventory: SdxlVisualInventory,
    prompts: SdxlVisualPromptSet,
) -> tuple[SdxlVisualCase, ...]:
    """Return causal controls and the three unresolved topology proofs."""

    base = prompts.control_case(
        "composed-lora-control",
        "Composed LoRA control",
        regional_prompt_weight=1.0,
    )
    style = RegionalVisualAdapter(
        STYLE_SELECTION,
        1.0,
        1.0,
        FULL_STRENGTH_SCHEDULE,
    )
    character = RegionalVisualAdapter(
        LEFT_CHARACTER_SELECTION,
        1.0,
        1.0,
        FULL_STRENGTH_SCHEDULE,
    )
    right_character = RegionalVisualAdapter(
        RIGHT_CHARACTER_SELECTION,
        1.0,
        1.0,
        FULL_STRENGTH_SCHEDULE,
    )
    styled_left_g = _prefix(
        inventory.style.prompt_g,
        inventory.left_character.prompt_g,
    )
    styled_left_l = _prefix(
        inventory.style.prompt_l,
        inventory.left_character.prompt_l,
    )
    global_style = replace(
        base,
        case_id="composed-global-style-control",
        label=f"{inventory.style.label} global 1.0; section controls",
        base_positive_g=_prefix(
            inventory.style.prompt_g,
            prompts.base_positive_g,
        ),
        base_positive_l=_prefix(
            inventory.style.prompt_l,
            prompts.base_positive_l,
        ),
        global_adapters=(GlobalVisualAdapter(STYLE_SELECTION, 1.0),),
    )
    repeated_global_style = replace(
        global_style,
        case_id="composed-same-style-global-control",
        label=(
            f"{inventory.style.label} global "
            f"{REPEATED_GLOBAL_STRENGTH}; repeated-use control"
        ),
        global_adapters=(
            GlobalVisualAdapter(STYLE_SELECTION, REPEATED_GLOBAL_STRENGTH),
        ),
    )
    return (
        replace(
            base,
            case_id="composed-multiple-left",
            label=(
                f"{inventory.left_character.label} and {inventory.style.label} "
                f"left 1.0 each; {inventory.right_character.label} right 1.0"
            ),
            left_g=styled_left_g,
            left_l=styled_left_l,
            right_g=inventory.right_character.prompt_g,
            right_l=inventory.right_character.prompt_l,
            left_adapters=(character, style),
            right_adapters=(right_character,),
        ),
        global_style,
        repeated_global_style,
        replace(
            repeated_global_style,
            case_id="composed-same-style-global-left",
            label=(
                f"{inventory.style.label} global {REPEATED_GLOBAL_STRENGTH} "
                f"and left regional {REPEATED_REGIONAL_STRENGTH}"
            ),
            left_adapters=(
                RegionalVisualAdapter(
                    STYLE_SELECTION,
                    REPEATED_REGIONAL_STRENGTH,
                    REPEATED_REGIONAL_STRENGTH,
                    FULL_STRENGTH_SCHEDULE,
                ),
            ),
        ),
        replace(
            base,
            case_id="composed-same-style-regional-control",
            label="No LoRA; bilateral regional-style product control",
        ),
        replace(
            base,
            case_id="composed-same-style-both-regions",
            label=f"{inventory.style.label} independently in both regions 1.0",
            left_g=_prefix(inventory.style.prompt_g, prompts.left_positive_g),
            left_l=_prefix(inventory.style.prompt_l, prompts.left_positive_l),
            right_g=_prefix(inventory.style.prompt_g, prompts.right_positive_g),
            right_l=_prefix(inventory.style.prompt_l, prompts.right_positive_l),
            left_adapters=(style,),
            right_adapters=(style,),
        ),
    )


def _prefix(prefix: str, prompt: str) -> str:
    """Place the adapter trigger before its authored subject prompt."""

    if not prefix.strip() or not prompt.strip():
        raise ValueError("Composed LoRA trigger and prompt must be non-empty.")
    return f"{prefix}, {prompt}"
