# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declare the two decisive post-optimization SDXL visual cases."""

from __future__ import annotations

from dataclasses import replace

from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    STYLE_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    GlobalVisualAdapter,
    SdxlVisualCase,
)
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_full_strength_lora_fidelity.composition_cases import (
    full_strength_composition_cases,
)
from tools.sdxl_two_character_oracle import two_character_oracle_cases


def post_optimization_visual_cases(
    inventory: SdxlVisualInventory,
    prompts: SdxlVisualPromptSet,
) -> tuple[SdxlVisualCase, SdxlVisualCase]:
    """Return simultaneous characters and global-style/right-character proof."""

    _, right_only, _ = full_strength_composition_cases(inventory, prompts)
    (simultaneous,) = two_character_oracle_cases(inventory, prompts)
    simultaneous_proof = replace(
        simultaneous,
        case_id="post-cache-simultaneous-characters",
        label=(
            f"POST-CACHE — {inventory.left_character.label} left 1.0; "
            f"{inventory.right_character.label} right 1.0"
        ),
    )
    global_style_right = replace(
        right_only,
        case_id="post-cache-global-style-right-character",
        label=(
            f"POST-CACHE — {inventory.style.label} global 1.0; pink-haired "
            f"left control; {inventory.right_character.label} right 1.0"
        ),
        base_positive_g=_prefix(
            inventory.style.prompt_g,
            right_only.base_positive_g,
        ),
        base_positive_l=_prefix(
            inventory.style.prompt_l,
            right_only.base_positive_l,
        ),
        global_adapters=(GlobalVisualAdapter(STYLE_SELECTION, 1.0),),
    )
    return simultaneous_proof, global_style_right


def _prefix(prefix: str, prompt: str) -> str:
    """Place one authored style trigger before the complete global prompt."""

    return ", ".join(value.strip() for value in (prefix, prompt) if value.strip())
