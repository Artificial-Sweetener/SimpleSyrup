# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own shared regional Attention Coupling contracts and validation."""

from __future__ import annotations

from enum import StrEnum

from .regional_lora_plan import RegionalLoraPlan
from .regional_mask_bank import RegionalMaskBank


class RegionalAttentionBranch(StrEnum):
    """Name the processed conditioning bank selected for one Comfy chunk."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


def require_non_negative_regional_attention_index(
    value: object,
    *,
    name: str,
) -> None:
    """Require one non-negative integer regional-attention index."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Regional attention {name} must be an integer.")
    if value < 0:
        raise ValueError(f"Regional attention {name} must be non-negative.")


def validate_regional_attention_plan_authorities(
    *,
    mask_bank: object,
    lora_plan: object,
    positive_region_count: int,
    negative_region_count: int,
) -> None:
    """Validate shared mask, LoRA, and branch-count plan authorities."""

    if not isinstance(mask_bank, RegionalMaskBank):
        raise TypeError("Regional attention plan requires a RegionalMaskBank.")
    if not isinstance(lora_plan, RegionalLoraPlan):
        raise TypeError("Regional attention plan requires a RegionalLoraPlan.")
    for branch_name, region_count in (
        ("positive", positive_region_count),
        ("negative", negative_region_count),
    ):
        require_non_negative_regional_attention_index(
            region_count,
            name=f"{branch_name} region count",
        )
        if region_count > mask_bank.region_count:
            raise ValueError(
                f"Regional attention {branch_name} branch exceeds the mask bank."
            )
    out_of_bounds = tuple(
        adapter
        for adapter in lora_plan.adapters
        if adapter.region_index >= mask_bank.region_count
    )
    if out_of_bounds:
        indices = ", ".join(str(adapter.region_index) for adapter in out_of_bounds)
        raise ValueError(
            "Regional attention LoRA region indices exceed the mask bank: " + indices
        )
