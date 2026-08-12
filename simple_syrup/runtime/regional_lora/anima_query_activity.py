# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve one shared projected activity plan for every patched Anima block."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

import torch

from ...domain.regional_attention_batch import BatchedRegionalAttentionContexts
from ...domain.regional_attention_weights import (
    RegionalAttentionWeightingPolicy,
    RegionalAttentionWeights,
)
from .anima_activation_context import AnimaActivationGeometry
from .anima_attention_execution import AnimaRegionalAttentionExecution
from .anima_branch_batch import (
    ANIMA_BASE_BRANCH_KEY,
    ANIMA_REGIONAL_BRANCH_BATCH_BUILDER,
    AnimaRegionalBranchBatch,
    AnimaRegionalBranchBatchBuilder,
    AnimaRegionalBranchKey,
)
from .anima_cross_attention_weights import (
    ANIMA_CROSS_ATTENTION_WEIGHTING_POLICY,
)
from .anima_query_mask_context import (
    ANIMA_QUERY_MASK_CONTEXT,
    AnimaQueryMaskContext,
)
from .anima_query_masks import AnimaQueryMaskBatch


@dataclass(frozen=True, slots=True)
class AnimaRegionalQueryActivity:
    """Retain shared masks, weights, and compact branch batches for one call."""

    masks: AnimaQueryMaskBatch
    attention_weights: RegionalAttentionWeights
    attention_branches: AnimaRegionalBranchBatch
    adaln_branches: AnimaRegionalBranchBatch


@dataclass(frozen=True, slots=True)
class _AnimaRegionalQueryActivitySlot:
    """Bind one immutable activity value to all authorities that produced it."""

    geometry: AnimaActivationGeometry
    execution: AnimaRegionalAttentionExecution
    active_contexts: BatchedRegionalAttentionContexts
    device: torch.device
    dtype: torch.dtype
    activity: AnimaRegionalQueryActivity


class AnimaRegionalQueryActivityContext:
    """Own task-local projected activity reuse across attention, AdaLN, and LoRA."""

    def __init__(
        self,
        query_masks: AnimaQueryMaskContext = ANIMA_QUERY_MASK_CONTEXT,
        weighting: RegionalAttentionWeightingPolicy | None = None,
        branch_batches: AnimaRegionalBranchBatchBuilder = (
            ANIMA_REGIONAL_BRANCH_BATCH_BUILDER
        ),
    ) -> None:
        """Retain focused projection, weighting, and branch-packing collaborators."""

        if not isinstance(query_masks, AnimaQueryMaskContext):
            raise TypeError("Anima query activity requires a mask context.")
        if not isinstance(branch_batches, AnimaRegionalBranchBatchBuilder):
            raise TypeError("Anima query activity requires a branch-batch builder.")
        self._query_masks = query_masks
        self._weighting = weighting or ANIMA_CROSS_ATTENTION_WEIGHTING_POLICY
        self._branch_batches = branch_batches
        self._slot: ContextVar[_AnimaRegionalQueryActivitySlot | None] = ContextVar(
            "simple_syrup_anima_regional_query_activity",
            default=None,
        )

    def resolve(
        self,
        execution: AnimaRegionalAttentionExecution,
        geometry: AnimaActivationGeometry,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> AnimaRegionalQueryActivity:
        """Return one activity plan for the exact current model invocation."""

        if not isinstance(execution, AnimaRegionalAttentionExecution):
            raise TypeError("Anima query activity requires an attention execution.")
        if not isinstance(geometry, AnimaActivationGeometry):
            raise TypeError("Anima query activity requires activation geometry.")
        target_device = torch.device(device)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("Anima query activity requires a floating dtype.")
        active_contexts = execution.active_contexts
        slot = self._slot.get()
        if (
            slot is not None
            and slot.geometry is geometry
            and slot.execution is execution
            and slot.active_contexts is active_contexts
            and slot.device == target_device
            and slot.dtype == dtype
        ):
            return slot.activity
        masks = self._query_masks.resolve(
            execution,
            geometry,
            device=target_device,
            dtype=dtype,
        )
        weights = self._weighting.weights(
            masks.flattened,
            region_strengths=execution.region_strengths,
        )
        activity = AnimaRegionalQueryActivity(
            masks,
            weights,
            self._attention_branches(execution, weights),
            self._adaln_branches(masks),
        )
        self._slot.set(
            _AnimaRegionalQueryActivitySlot(
                geometry,
                execution,
                active_contexts,
                target_device,
                dtype,
                activity,
            )
        )
        return activity

    def clear(self) -> None:
        """Release the current task's projected activity tensors."""

        self._slot.set(None)

    def _attention_branches(
        self,
        execution: AnimaRegionalAttentionExecution,
        weights: RegionalAttentionWeights,
    ) -> AnimaRegionalBranchBatch:
        """Build exact prompt-entry support from normalized attention weights."""

        supports: list[tuple[AnimaRegionalBranchKey, torch.Tensor]] = [
            (
                ANIMA_BASE_BRANCH_KEY,
                weights.base.ne(0).flatten(start_dim=1).any(dim=1),
            )
        ]
        for region in execution.active_contexts.regions:
            regional_support = (
                weights.regions[region.region_index]
                .ne(0)
                .flatten(start_dim=1)
                .any(dim=1)
            )
            for entry in region.entries:
                supports.append(
                    (
                        AnimaRegionalBranchKey(
                            region.region_index,
                            entry.entry_index,
                        ),
                        regional_support
                        & weights.base.new_tensor(entry.strengths).ne(0),
                    )
                )
        return self._branch_batches.build(
            source_batch_size=int(weights.base.shape[0]),
            supports=tuple(supports),
        )

    def _adaln_branches(
        self,
        masks: AnimaQueryMaskBatch,
    ) -> AnimaRegionalBranchBatch:
        """Build exact AdaLN support from its established mask-difference formula."""

        raw_masks = masks.masks
        supports: list[tuple[AnimaRegionalBranchKey, torch.Tensor]] = [
            (
                ANIMA_BASE_BRANCH_KEY,
                (1.0 - raw_masks.sum(dim=0)).ne(0).flatten(start_dim=1).any(dim=1),
            )
        ]
        supports.extend(
            (
                AnimaRegionalBranchKey(region_index, 0),
                raw_masks[region_index].ne(0).flatten(start_dim=1).any(dim=1),
            )
            for region_index in range(int(raw_masks.shape[0]))
        )
        return self._branch_batches.build(
            source_batch_size=int(raw_masks.shape[1]),
            supports=tuple(supports),
        )


ANIMA_REGIONAL_QUERY_ACTIVITY_CONTEXT = AnimaRegionalQueryActivityContext()
