# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure policies for global-first regional prompt pairing."""

from __future__ import annotations

import math
from dataclasses import dataclass

MAX_REGIONAL_PROMPT_WEIGHT = 1.0


@dataclass(frozen=True)
class RegionalConditioningPair:
    """Map one regional conditioning entry to its authored mask index."""

    conditioning_index: int
    mask_index: int


@dataclass(frozen=True)
class RegionalConditioningPlan:
    """Describe valid positional pairing after the global entry."""

    region_count: int
    pairs: tuple[RegionalConditioningPair, ...]


def validate_regional_prompt_weight(weight: float) -> None:
    """Reject regional influence values outside the normalized blend range."""

    if not math.isfinite(weight):
        raise ValueError("regional_prompt_weight must be finite.")
    if not 0.0 <= weight <= MAX_REGIONAL_PROMPT_WEIGHT:
        raise ValueError(
            "regional_prompt_weight must be between 0.0 and "
            f"{MAX_REGIONAL_PROMPT_WEIGHT:.1f}."
        )


def build_regional_conditioning_plan(
    *,
    region_count: int,
    conditioning_count: int,
    input_name: str,
) -> RegionalConditioningPlan:
    """Return the global-first positional plan or reject excess prompts."""

    if region_count < 1:
        raise ValueError("regional prompting requires at least one authored mask.")
    if conditioning_count < 1:
        raise ValueError(
            f"{input_name} conditioning must contain a global entry at index 0."
        )

    regional_count = conditioning_count - 1
    if regional_count > region_count:
        raise ValueError(
            f"{input_name} conditioning contains {regional_count} regional "
            f"entries but only {region_count} authored masks were provided."
        )

    return RegionalConditioningPlan(
        region_count=region_count,
        pairs=tuple(
            RegionalConditioningPair(
                conditioning_index=mask_index + 1,
                mask_index=mask_index,
            )
            for mask_index in range(regional_count)
        ),
    )
