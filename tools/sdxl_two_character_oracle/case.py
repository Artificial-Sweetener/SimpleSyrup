# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own the accepted full-strength SDXL two-character oracle case."""

from __future__ import annotations

from dataclasses import replace

from tools.sdxl_attention_coupling_integration.visual_case_model import (
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

SDXL_TWO_CHARACTER_ORACLE_ID = "sdxl-two-character-oracle"


def two_character_oracle_cases(
    inventory: SdxlVisualInventory,
    prompts: SdxlVisualPromptSet,
) -> tuple[SdxlVisualCase, ...]:
    """Return the accepted full-strength left/right character composition."""

    simultaneous = full_strength_composition_cases(inventory, prompts)[-1]
    return (
        replace(
            simultaneous,
            case_id=SDXL_TWO_CHARACTER_ORACLE_ID,
            label=(
                f"ORACLE — {inventory.left_character.label} left 1.0; "
                f"{inventory.right_character.label} right 1.0"
            ),
        ),
    )
