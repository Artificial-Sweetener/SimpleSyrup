# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Classify Anima LoRA target families by multiplier execution scope."""

from __future__ import annotations

from enum import StrEnum

from .anima_targets import AnimaLoraTargetFamily


class AnimaLoraWeightCategory(StrEnum):
    """Name each authoritative multiplier scope."""

    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    MLP = "mlp"
    ADALN = "adaln"


_CATEGORY_BY_FAMILY = {
    AnimaLoraTargetFamily.SELF_ATTN_Q: AnimaLoraWeightCategory.SELF_ATTENTION,
    AnimaLoraTargetFamily.SELF_ATTN_K: AnimaLoraWeightCategory.SELF_ATTENTION,
    AnimaLoraTargetFamily.SELF_ATTN_V: AnimaLoraWeightCategory.SELF_ATTENTION,
    AnimaLoraTargetFamily.SELF_ATTN_OUTPUT: AnimaLoraWeightCategory.SELF_ATTENTION,
    AnimaLoraTargetFamily.CROSS_ATTN_Q: AnimaLoraWeightCategory.CROSS_ATTENTION,
    AnimaLoraTargetFamily.CROSS_ATTN_K: AnimaLoraWeightCategory.CROSS_ATTENTION,
    AnimaLoraTargetFamily.CROSS_ATTN_V: AnimaLoraWeightCategory.CROSS_ATTENTION,
    AnimaLoraTargetFamily.CROSS_ATTN_OUTPUT: AnimaLoraWeightCategory.CROSS_ATTENTION,
    AnimaLoraTargetFamily.MLP_LAYER1: AnimaLoraWeightCategory.MLP,
    AnimaLoraTargetFamily.MLP_LAYER2: AnimaLoraWeightCategory.MLP,
    AnimaLoraTargetFamily.ADALN_SELF_ATTN_1: AnimaLoraWeightCategory.ADALN,
    AnimaLoraTargetFamily.ADALN_SELF_ATTN_2: AnimaLoraWeightCategory.ADALN,
    AnimaLoraTargetFamily.ADALN_CROSS_ATTN_1: AnimaLoraWeightCategory.ADALN,
    AnimaLoraTargetFamily.ADALN_CROSS_ATTN_2: AnimaLoraWeightCategory.ADALN,
    AnimaLoraTargetFamily.ADALN_MLP_1: AnimaLoraWeightCategory.ADALN,
    AnimaLoraTargetFamily.ADALN_MLP_2: AnimaLoraWeightCategory.ADALN,
}


def anima_lora_weight_category(
    family: AnimaLoraTargetFamily,
) -> AnimaLoraWeightCategory:
    """Return the complete multiplier category for one admitted family."""

    if not isinstance(family, AnimaLoraTargetFamily):
        raise TypeError("Anima LoRA weight category requires a target family.")
    try:
        return _CATEGORY_BY_FAMILY[family]
    except KeyError as error:
        raise ValueError(f"Unsupported Anima LoRA weight family {family!r}.") from error
