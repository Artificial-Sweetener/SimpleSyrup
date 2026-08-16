# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build the bounded external-prompt SDXL LoRA baseline matrix."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .visual_adapter_controls import (
    global_style_adapter,
    left_character_adapter,
    right_character_adapter,
)
from .visual_case_model import SdxlVisualCase
from .visual_inventory import SdxlVisualInventory


@dataclass(frozen=True, slots=True)
class SdxlVisualPromptSet:
    """Retain one complete external global/left/right SEP prompt fixture."""

    base_positive_g: str
    base_positive_l: str
    base_negative_g: str
    base_negative_l: str
    left_positive_g: str
    left_positive_l: str
    right_positive_g: str
    right_positive_l: str
    left_negative_g: str
    left_negative_l: str
    right_negative_g: str
    right_negative_l: str

    def control_case(
        self,
        case_id: str,
        label: str,
        *,
        regional_prompt_weight: float = 1.0,
    ) -> SdxlVisualCase:
        """Materialize one exact no-LoRA section-control declaration."""

        return SdxlVisualCase(
            case_id,
            label,
            base_positive_l=self.base_positive_l,
            base_positive_g=self.base_positive_g,
            base_negative_l=self.base_negative_l,
            base_negative_g=self.base_negative_g,
            left_l=self.left_positive_l,
            left_g=self.left_positive_g,
            right_l=self.right_positive_l,
            right_g=self.right_positive_g,
            left_negative_l=self.left_negative_l,
            left_negative_g=self.left_negative_g,
            right_negative_l=self.right_negative_l,
            right_negative_g=self.right_negative_g,
            regional_prompt_weight=regional_prompt_weight,
        )


def lora_baseline_cases(
    inventory: SdxlVisualInventory,
    prompts: SdxlVisualPromptSet,
) -> tuple[SdxlVisualCase, ...]:
    """Return left-only, right-only, simultaneous, and global-style cases."""

    left_control = prompts.control_case("left-character-baseline", "")
    right_control = prompts.control_case("right-character-baseline", "")
    simultaneous_control = prompts.control_case(
        "simultaneous-characters-baseline",
        "",
    )
    style_control = prompts.control_case("global-style-baseline", "")
    return (
        replace(
            left_control,
            label=f"{inventory.left_character.label} left; right section control",
            left_l=inventory.left_character.prompt_l,
            left_g=inventory.left_character.prompt_g,
            left_adapters=(left_character_adapter(),),
        ),
        replace(
            right_control,
            label=f"Left section control; {inventory.right_character.label} right",
            right_l=inventory.right_character.prompt_l,
            right_g=inventory.right_character.prompt_g,
            right_adapters=(right_character_adapter(),),
        ),
        replace(
            simultaneous_control,
            label=(
                f"{inventory.left_character.label} left; "
                f"{inventory.right_character.label} right"
            ),
            left_l=inventory.left_character.prompt_l,
            left_g=inventory.left_character.prompt_g,
            right_l=inventory.right_character.prompt_l,
            right_g=inventory.right_character.prompt_g,
            left_adapters=(left_character_adapter(),),
            right_adapters=(right_character_adapter(),),
        ),
        replace(
            style_control,
            label=f"Global {inventory.style.label}; both section controls",
            base_positive_l=_prefix(
                inventory.style.prompt_l,
                prompts.base_positive_l,
            ),
            base_positive_g=_prefix(
                inventory.style.prompt_g,
                prompts.base_positive_g,
            ),
            global_adapters=(global_style_adapter(),),
        ),
    )


def _prefix(prefix: str, prompt: str) -> str:
    """Place one style trigger before the authored global prompt."""

    if not prefix.strip() or not prompt.strip():
        raise ValueError("Global style prefix and prompt must be non-empty.")
    return f"{prefix}, {prompt}"
