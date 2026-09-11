# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own immutable processed regional Attention Coupling runtime plans."""

from __future__ import annotations

import math
from dataclasses import dataclass
from uuid import UUID

import torch

from .conditioning_schedule import ConditioningScheduleRange
from .regional_attention import (
    require_non_negative_regional_attention_index,
    validate_regional_attention_plan_authorities,
)
from .regional_lora_plan import RegionalLoraPlan
from .regional_mask_bank import RegionalMaskBank


@dataclass(frozen=True, slots=True)
class ProcessedRegionalAttentionEntry:
    """Retain one ordered model-ready conditioning entry and Comfy strength."""

    entry_index: int
    uuid: UUID
    schedule: ConditioningScheduleRange
    cross_attention: torch.Tensor
    strength: float
    cross_attention_value_multiplier: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate entry order, model context, and finite scalar strength."""

        require_non_negative_regional_attention_index(
            self.entry_index,
            name="entry_index",
        )
        if not isinstance(self.uuid, UUID):
            raise TypeError("Processed conditioning entry UUID must be uuid.UUID.")
        if not isinstance(self.schedule, ConditioningScheduleRange):
            raise TypeError(
                "Processed conditioning entry schedule has an invalid type."
            )
        if not isinstance(self.cross_attention, torch.Tensor):
            raise TypeError("Processed cross_attention must be a torch.Tensor.")
        if self.cross_attention.ndim != 3:
            raise ValueError("Processed cross_attention must use BxSxD layout.")
        if any(int(size) < 1 for size in self.cross_attention.shape):
            raise ValueError("Processed cross_attention dimensions must be positive.")
        if not self.cross_attention.is_floating_point():
            raise TypeError("Processed cross_attention must be floating point.")
        if not bool(torch.isfinite(self.cross_attention).all().item()):
            raise ValueError("Processed cross_attention must contain finite values.")
        if isinstance(self.strength, bool) or not isinstance(
            self.strength,
            int | float,
        ):
            raise TypeError("Processed conditioning strength must be a real number.")
        if not math.isfinite(float(self.strength)):
            raise ValueError("Processed conditioning strength must be finite.")
        object.__setattr__(self, "strength", float(self.strength))
        multiplier = self.cross_attention_value_multiplier
        if multiplier is None:
            return
        if (
            not isinstance(multiplier, torch.Tensor)
            or multiplier.shape != (*self.cross_attention.shape[:2], 1)
            or not multiplier.is_floating_point()
            or multiplier.device != self.cross_attention.device
            or multiplier.dtype != self.cross_attention.dtype
            or not bool(torch.isfinite(multiplier).all().item())
        ):
            raise ValueError(
                "Processed attention value multiplier must be a finite floating "
                "BxSx1 tensor aligned with cross_attention."
            )


@dataclass(frozen=True, slots=True)
class ProcessedRegionalAttentionContext:
    """Retain every ordered processed entry for one authored conditioning."""

    conditioning_index: int
    region_index: int | None
    entries: tuple[ProcessedRegionalAttentionEntry, ...]

    def __post_init__(self) -> None:
        """Validate global/regional ownership and model-ready tensor structure."""

        if self.region_index is None:
            if self.conditioning_index != 0:
                raise ValueError(
                    "Processed base attention context must use conditioning index 0."
                )
        else:
            require_non_negative_regional_attention_index(
                self.region_index,
                name="region_index",
            )
            if self.conditioning_index != self.region_index + 1:
                raise ValueError(
                    "Processed regional conditioning_index must equal region_index + 1."
                )
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("Processed attention context requires ordered entries.")
        if any(
            not isinstance(entry, ProcessedRegionalAttentionEntry)
            for entry in self.entries
        ):
            raise TypeError("Processed attention context contains an invalid entry.")
        if tuple(entry.entry_index for entry in self.entries) != tuple(
            range(len(self.entries))
        ):
            raise ValueError("Processed attention entries must use canonical order.")


@dataclass(frozen=True, slots=True)
class ProcessedRegionalAttentionBranch:
    """Retain one processed base context and ordered regional context bank."""

    base_context: ProcessedRegionalAttentionContext
    regional_contexts: tuple[ProcessedRegionalAttentionContext, ...]

    def __post_init__(self) -> None:
        """Require one global base and canonical regional context order."""

        if not isinstance(self.base_context, ProcessedRegionalAttentionContext):
            raise TypeError("Processed attention base context has an invalid type.")
        if self.base_context.region_index is not None:
            raise ValueError("Processed attention base context must be global.")
        if not isinstance(self.regional_contexts, tuple):
            raise TypeError("Processed regional attention contexts must be a tuple.")
        if any(
            not isinstance(context, ProcessedRegionalAttentionContext)
            for context in self.regional_contexts
        ):
            raise TypeError(
                "Processed regional attention branch contains an invalid context."
            )
        indices = tuple(context.region_index for context in self.regional_contexts)
        if indices != tuple(range(len(self.regional_contexts))):
            raise ValueError(
                "Processed regional attention contexts must use canonical order."
            )


@dataclass(frozen=True, slots=True)
class ProcessedRegionalAttentionPlan:
    """Retain processed branches and shared canonical regional authorities."""

    positive: ProcessedRegionalAttentionBranch
    negative: ProcessedRegionalAttentionBranch
    mask_bank: RegionalMaskBank
    lora_plan: RegionalLoraPlan

    def __post_init__(self) -> None:
        """Validate plan owners and regional LoRA bounds."""

        if not isinstance(
            self.positive, ProcessedRegionalAttentionBranch
        ) or not isinstance(self.negative, ProcessedRegionalAttentionBranch):
            raise TypeError(
                "Processed regional attention plan contains an invalid branch."
            )
        validate_regional_attention_plan_authorities(
            mask_bank=self.mask_bank,
            lora_plan=self.lora_plan,
            positive_region_count=len(self.positive.regional_contexts),
            negative_region_count=len(self.negative.regional_contexts),
        )

    @property
    def is_time_invariant(self) -> bool:
        """Report whether contexts and regional LoRA strengths remain fixed."""

        branches = (self.positive, self.negative)
        contexts = tuple(
            context
            for branch in branches
            for context in (branch.base_context, *branch.regional_contexts)
        )
        return self.lora_plan.is_time_invariant and all(
            entry.schedule.is_time_invariant
            for context in contexts
            for entry in context.entries
        )
