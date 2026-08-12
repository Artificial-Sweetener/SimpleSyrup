# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own immutable raw regional Attention Coupling authoring plans."""

from __future__ import annotations

from dataclasses import dataclass

from .conditioning_batch import ConditioningBatch
from .regional_attention import (
    require_non_negative_regional_attention_index,
    validate_regional_attention_plan_authorities,
)
from .regional_lora_plan import EMPTY_REGIONAL_LORA_PLAN, RegionalLoraPlan
from .regional_mask_bank import RegionalMaskBank
from .regional_prompting import build_regional_conditioning_plan


@dataclass(frozen=True, slots=True)
class RawRegionalAttentionContext:
    """Retain one original regional conditioning and its canonical indices."""

    conditioning_index: int
    region_index: int
    conditioning: object

    def __post_init__(self) -> None:
        """Require the established global-first positional relationship."""

        require_non_negative_regional_attention_index(
            self.region_index,
            name="region_index",
        )
        if self.conditioning_index != self.region_index + 1:
            raise ValueError(
                "Regional attention conditioning_index must equal region_index + 1."
            )


@dataclass(frozen=True, slots=True)
class RawRegionalAttentionBranch:
    """Retain one base conditioning and ordered regional conditionings."""

    base_conditioning: object
    regional_contexts: tuple[RawRegionalAttentionContext, ...]

    def __post_init__(self) -> None:
        """Require immutable canonical regional order."""

        if not isinstance(self.regional_contexts, tuple):
            raise TypeError("Raw regional attention contexts must be a tuple.")
        if any(
            not isinstance(context, RawRegionalAttentionContext)
            for context in self.regional_contexts
        ):
            raise TypeError(
                "Raw regional attention branch contains an invalid context."
            )
        indices = tuple(context.region_index for context in self.regional_contexts)
        if indices != tuple(range(len(self.regional_contexts))):
            raise ValueError(
                "Raw regional attention contexts must use canonical order."
            )


@dataclass(frozen=True, slots=True)
class RawRegionalAttentionPlan:
    """Retain both raw branches and shared canonical regional authorities."""

    positive: RawRegionalAttentionBranch
    negative: RawRegionalAttentionBranch
    mask_bank: RegionalMaskBank
    lora_plan: RegionalLoraPlan

    def __post_init__(self) -> None:
        """Validate plan owners and regional LoRA bounds."""

        if not isinstance(self.positive, RawRegionalAttentionBranch) or not isinstance(
            self.negative, RawRegionalAttentionBranch
        ):
            raise TypeError("Raw regional attention plan contains an invalid branch.")
        validate_regional_attention_plan_authorities(
            mask_bank=self.mask_bank,
            lora_plan=self.lora_plan,
            positive_region_count=len(self.positive.regional_contexts),
            negative_region_count=len(self.negative.regional_contexts),
        )


def build_raw_regional_attention_plan(
    *,
    positive: object,
    negative: object,
    mask_bank: RegionalMaskBank,
    lora_plan: RegionalLoraPlan = EMPTY_REGIONAL_LORA_PLAN,
) -> RawRegionalAttentionPlan:
    """Build both branches through the authoritative global-first pairing policy."""

    if not isinstance(mask_bank, RegionalMaskBank):
        raise TypeError("Raw regional attention requires a RegionalMaskBank.")
    return RawRegionalAttentionPlan(
        positive=_build_raw_branch(
            positive,
            mask_bank=mask_bank,
            input_name="positive",
        ),
        negative=_build_raw_branch(
            negative,
            mask_bank=mask_bank,
            input_name="negative",
        ),
        mask_bank=mask_bank,
        lora_plan=lora_plan,
    )


def _build_raw_branch(
    conditioning: object,
    *,
    mask_bank: RegionalMaskBank,
    input_name: str,
) -> RawRegionalAttentionBranch:
    """Pair one raw conditioning branch without restating index policy."""

    entries = (
        conditioning.entries
        if isinstance(conditioning, ConditioningBatch)
        else (conditioning,)
    )
    pairing = build_regional_conditioning_plan(
        region_count=mask_bank.region_count,
        conditioning_count=len(entries),
        input_name=input_name,
    )
    return RawRegionalAttentionBranch(
        base_conditioning=entries[0],
        regional_contexts=tuple(
            RawRegionalAttentionContext(
                conditioning_index=pair.conditioning_index,
                region_index=pair.mask_index,
                conditioning=entries[pair.conditioning_index],
            )
            for pair in pairing.pairs
        ),
    )
