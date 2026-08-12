# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cache one exact compact Anima attention plan across transformer blocks."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

import torch
from comfy.patcher_extension import CallbacksMP

from ..model_patcher_mutations import ModelKeyedCallbackMutation
from .anima_self_attention_coherence import (
    ANIMA_SELF_ATTENTION_COHERENCE_POLICY,
    AnimaSelfAttentionCoherencePolicy,
)
from .anima_self_attention_ownership import (
    ANIMA_SELF_ATTENTION_OWNERSHIP_POLICY,
    AnimaSelfAttentionOwnershipPolicy,
)
from .anima_self_attention_partition import AnimaSelfAttentionPartitionPlan

_CACHE_CALLBACK_KEY = "simple_syrup.anima_self_attention_partition_cache"


@dataclass(frozen=True, slots=True)
class _PartitionPlanSlot:
    """Bind one projected mask identity and geometry to its compact plan."""

    masks: torch.Tensor
    height: int
    width: int
    plan: AnimaSelfAttentionPartitionPlan


class AnimaSelfAttentionPartitionCache:
    """Build and retain one compact attention plan for the active model call."""

    def __init__(
        self,
        ownership: AnimaSelfAttentionOwnershipPolicy = (
            ANIMA_SELF_ATTENTION_OWNERSHIP_POLICY
        ),
        coherence: AnimaSelfAttentionCoherencePolicy = (
            ANIMA_SELF_ATTENTION_COHERENCE_POLICY
        ),
    ) -> None:
        """Retain focused policies and initialize an empty task-local slot."""

        if not isinstance(ownership, AnimaSelfAttentionOwnershipPolicy):
            raise TypeError("Anima partition cache requires an ownership policy.")
        if not isinstance(coherence, AnimaSelfAttentionCoherencePolicy):
            raise TypeError("Anima partition cache requires a coherence policy.")
        self._ownership = ownership
        self._coherence = coherence
        self._slot: ContextVar[_PartitionPlanSlot | None] = ContextVar(
            "simple_syrup_anima_self_attention_partition_cache", default=None
        )

    def resolve(
        self,
        masks: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> AnimaSelfAttentionPartitionPlan:
        """Return the exact cached plan for one projected mask batch and grid."""

        slot = self._slot.get()
        if (
            slot is not None
            and slot.masks is masks
            and slot.height == height
            and slot.width == width
        ):
            return slot.plan
        profile = self._coherence.resolve(
            self._ownership.token_owners(masks),
            height=height,
            width=width,
        )
        plan = AnimaSelfAttentionPartitionPlan.build(profile)
        self._slot.set(_PartitionPlanSlot(masks, height, width, plan))
        return plan

    def clear(self, model: object, unpatch_all: bool) -> None:
        """Release the current task's retained device partition indices."""

        del model, unpatch_all
        self._slot.set(None)

    def mutation(self) -> ModelKeyedCallbackMutation:
        """Return the clone-lifecycle callback that owns cache cleanup."""

        return ModelKeyedCallbackMutation(
            CallbacksMP.ON_DETACH,
            _CACHE_CALLBACK_KEY,
            self.clear,
        )
