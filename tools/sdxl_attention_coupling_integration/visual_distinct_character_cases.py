# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own the controlled distinct-character SDXL visual case family."""

from __future__ import annotations

from .visual_case_model import RegionalVisualAdapter, SdxlVisualCase
from .visual_inventory import SdxlVisualInventory


def distinct_character_cases(
    inventory: SdxlVisualInventory,
    *,
    left_adapter: RegionalVisualAdapter,
    right_adapter: RegionalVisualAdapter,
) -> tuple[SdxlVisualCase, ...]:
    """Return prompt-only, left-only, and simultaneous parity-baseline cases."""

    return (
        SdxlVisualCase(
            "different-characters-prompt-control",
            "Different character prompts without LoRAs",
            left_l=inventory.left_character.prompt_l,
            left_g=inventory.left_character.prompt_g,
            right_l=inventory.right_character.prompt_l,
            right_g=inventory.right_character.prompt_g,
        ),
        SdxlVisualCase(
            "different-characters-left-adapter-control",
            "Left character adapter with distinct right prompt",
            left_l=inventory.left_character.prompt_l,
            left_g=inventory.left_character.prompt_g,
            right_l=inventory.right_character.prompt_l,
            right_g=inventory.right_character.prompt_g,
            left_adapters=(left_adapter,),
        ),
        SdxlVisualCase(
            "different-characters",
            "Different character adapters on left and right",
            left_l=inventory.left_character.prompt_l,
            left_g=inventory.left_character.prompt_g,
            right_l=inventory.right_character.prompt_l,
            right_g=inventory.right_character.prompt_g,
            left_adapters=(left_adapter,),
            right_adapters=(right_adapter,),
        ),
    )
