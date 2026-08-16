# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declare identical-prompt zero/one/four SDXL regional scaling cases."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    LEFT_CHARACTER_SELECTION,
    RIGHT_CHARACTER_SELECTION,
    STYLE_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    RegionalVisualAdapter,
    SdxlVisualCase,
)
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)

FULL_STRENGTH_SCHEDULE = ((0.0, 1.0),)


class SdxlRegionalScalingProfile(StrEnum):
    """Identify the required active adapter-use cardinalities."""

    ZERO = "zero-active-uses"
    ONE = "one-active-use"
    FOUR = "four-active-uses"


@dataclass(frozen=True, slots=True)
class DeclaredSdxlRegionalScalingCase:
    """Pair one scaling profile with its complete visual case."""

    profile: SdxlRegionalScalingProfile
    case: SdxlVisualCase

    @property
    def adapter_count(self) -> int:
        """Return the exact authored regional adapter-use count."""

        return len(self.case.left_adapters) + len(self.case.right_adapters)


def regional_scaling_cases(
    prompts: SdxlVisualPromptSet,
    *,
    left_trigger_g: str,
    left_trigger_l: str,
    right_trigger_g: str,
    right_trigger_l: str,
    style_trigger_g: str,
    style_trigger_l: str,
) -> tuple[DeclaredSdxlRegionalScalingCase, ...]:
    """Return identical-prompt zero, one, and four-use declarations."""

    common = replace(
        prompts.control_case("regional-scaling", "Regional scaling"),
        left_g=_join(style_trigger_g, left_trigger_g),
        left_l=_join(style_trigger_l, left_trigger_l),
        right_g=_join(style_trigger_g, right_trigger_g),
        right_l=_join(style_trigger_l, right_trigger_l),
    )
    left_character = _adapter(LEFT_CHARACTER_SELECTION)
    right_character = _adapter(RIGHT_CHARACTER_SELECTION)
    style = _adapter(STYLE_SELECTION)
    return (
        DeclaredSdxlRegionalScalingCase(
            SdxlRegionalScalingProfile.ZERO,
            replace(common, case_id=SdxlRegionalScalingProfile.ZERO.value),
        ),
        DeclaredSdxlRegionalScalingCase(
            SdxlRegionalScalingProfile.ONE,
            replace(
                common,
                case_id=SdxlRegionalScalingProfile.ONE.value,
                left_adapters=(left_character,),
            ),
        ),
        DeclaredSdxlRegionalScalingCase(
            SdxlRegionalScalingProfile.FOUR,
            replace(
                common,
                case_id=SdxlRegionalScalingProfile.FOUR.value,
                left_adapters=(left_character, style),
                right_adapters=(right_character, style),
            ),
        ),
    )


def _adapter(name: str) -> RegionalVisualAdapter:
    """Return one full-strength all-step regional adapter use."""

    return RegionalVisualAdapter(name, 1.0, 1.0, FULL_STRENGTH_SCHEDULE)


def _join(*segments: str) -> str:
    """Join required non-empty scaling prompt segments."""

    normalized = tuple(segment.strip() for segment in segments)
    if any(not segment for segment in normalized):
        raise ValueError("Scaling trigger prompts must be non-empty.")
    return ", ".join(normalized)
