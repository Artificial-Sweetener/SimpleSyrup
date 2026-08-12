# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own one preprojected standard-UNet attn2 coupling execution."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ...domain.regional_attention_batch import BatchedRegionalAttentionContexts
from ...domain.regional_attention_weights import (
    RegionalAttentionWeightingPolicy,
    RegionalAttentionWeights,
)
from ...domain.regional_conditioning_output import (
    REGIONAL_CONDITIONING_OUTPUT_COMBINER,
    RegionalConditioningOutputCombiner,
)
from .unet_branch_batch import (
    UNET_ATTENTION_BRANCH_BATCH_BUILDER,
    UNET_BASE_ATTENTION_BRANCH,
    UnetAttentionBranchBatch,
    UnetAttentionBranchKey,
)


@dataclass(frozen=True, slots=True)
class ExpandedUnetAttn2Inputs:
    """Retain the exact packed tensors returned from an attn2 input patch."""

    query: torch.Tensor
    context: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True, slots=True)
class UnetAttn2Execution:
    """Bind aligned contexts to one exact BxQ projected regional mask bank."""

    contexts: BatchedRegionalAttentionContexts
    query_masks: torch.Tensor
    region_strengths: tuple[float, ...]
    weights: RegionalAttentionWeights = field(init=False)
    branches: UnetAttentionBranchBatch = field(init=False)

    def __post_init__(self) -> None:
        """Validate one query contract and precompute its reusable branch plan."""

        if not isinstance(self.contexts, BatchedRegionalAttentionContexts):
            raise TypeError("UNet attn2 execution requires aligned contexts.")
        if not isinstance(self.query_masks, torch.Tensor):
            raise TypeError("UNet attn2 query masks must be a tensor.")
        if (
            self.query_masks.ndim != 3
            or int(self.query_masks.shape[0]) < 1
            or any(int(size) < 1 for size in self.query_masks.shape[1:])
        ):
            raise ValueError("UNet attn2 query masks must use non-empty R/B/Q layout.")
        if not self.query_masks.is_floating_point():
            raise TypeError("UNet attn2 query masks must use a floating dtype.")
        if not bool(torch.isfinite(self.query_masks).all()):
            raise ValueError("UNet attn2 query masks must contain finite values.")
        if int(self.query_masks.shape[0]) != len(self.contexts.regions):
            raise ValueError(
                "UNet attn2 query-mask region count must match aligned contexts."
            )
        if int(self.query_masks.shape[1]) != int(self.contexts.base_context.shape[0]):
            raise ValueError("UNet attn2 query-mask batch must match aligned contexts.")
        if self.query_masks.device != self.contexts.base_context.device:
            raise ValueError("UNet attn2 masks and contexts must share one device.")
        weighting = RegionalAttentionWeightingPolicy()
        weights = weighting.weights(
            self.query_masks,
            region_strengths=self.region_strengths,
        )
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "branches", self._build_branches(weights))

    def expand(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        value: torch.Tensor,
    ) -> ExpandedUnetAttn2Inputs:
        """Pack supported base and regional rows for one native attention call."""

        self._validate_inputs(query, context, value)
        context_values = self._branch_context_values(context)
        value_values = dict(context_values)
        value_values[UNET_BASE_ATTENTION_BRANCH] = value
        return ExpandedUnetAttn2Inputs(
            self.branches.pack_source(query),
            self.branches.pack_branch_values(context_values),
            self.branches.pack_branch_values(value_values),
        )

    def blend(self, packed_output: torch.Tensor) -> torch.Tensor:
        """Restore, combine, and spatially blend complete attention outputs."""

        restored = self.branches.restore(packed_output)
        source_shape = (self.branches.source_batch_size, *packed_output.shape[1:])
        empty = packed_output.new_zeros(source_shape)
        base_output = restored.get(UNET_BASE_ATTENTION_BRANCH, empty)
        combiner: RegionalConditioningOutputCombiner = (
            REGIONAL_CONDITIONING_OUTPUT_COMBINER
        )
        regional_outputs = tuple(
            combiner.combine(
                tuple(
                    restored.get(
                        UnetAttentionBranchKey(
                            region.region_index,
                            entry.entry_index,
                        ),
                        empty,
                    )
                    for entry in region.entries
                ),
                strengths=tuple(entry.strengths for entry in region.entries),
            )
            for region in self.contexts.regions
        )
        return RegionalAttentionWeightingPolicy().blend(
            weights=self.weights,
            base_output=base_output,
            regional_outputs=torch.stack(regional_outputs),
        )

    def _build_branches(
        self,
        weights: RegionalAttentionWeights,
    ) -> UnetAttentionBranchBatch:
        """Build exact row support from normalized spatial and entry strengths."""

        supports: list[tuple[UnetAttentionBranchKey, torch.Tensor]] = [
            (
                UNET_BASE_ATTENTION_BRANCH,
                weights.base.ne(0).flatten(start_dim=1).any(dim=1),
            )
        ]
        for region in self.contexts.regions:
            spatial_support = (
                weights.regions[region.region_index]
                .ne(0)
                .flatten(start_dim=1)
                .any(dim=1)
            )
            for entry in region.entries:
                supports.append(
                    (
                        UnetAttentionBranchKey(
                            region.region_index,
                            entry.entry_index,
                        ),
                        spatial_support
                        & weights.base.new_tensor(entry.strengths).ne(0),
                    )
                )
        return UNET_ATTENTION_BRANCH_BATCH_BUILDER.build(
            source_batch_size=int(weights.base.shape[0]),
            supports=tuple(supports),
        )

    def _branch_context_values(
        self,
        base_context: torch.Tensor,
    ) -> dict[UnetAttentionBranchKey, torch.Tensor]:
        """Return exact callback base and plan-owned regional context tensors."""

        values = {UNET_BASE_ATTENTION_BRANCH: base_context}
        for region in self.contexts.regions:
            for entry in region.entries:
                values[
                    UnetAttentionBranchKey(region.region_index, entry.entry_index)
                ] = entry.context
        return values

    def _validate_inputs(
        self,
        query: object,
        context: object,
        value: object,
    ) -> None:
        """Require the exact source batch, query grid, and active base context."""

        if not isinstance(query, torch.Tensor) or query.ndim != 3:
            raise ValueError("UNet attn2 query must use BxQxD layout.")
        if not isinstance(context, torch.Tensor) or context.ndim != 3:
            raise ValueError("UNet attn2 context must use BxSxD layout.")
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError("UNet attn2 value must use BxSxD layout.")
        expected_batch = int(self.contexts.base_context.shape[0])
        if int(query.shape[0]) != expected_batch or int(query.shape[1]) != int(
            self.query_masks.shape[2]
        ):
            raise ValueError("UNet attn2 query does not match execution geometry.")
        if context is not self.contexts.base_context:
            raise ValueError(
                "UNet attn2 context must be the exact aligned base authority."
            )
        if value is not context:
            raise ValueError(
                "UNet attn2 value must be Comfy's exact pre-patch context object."
            )
        if query.device != context.device or query.dtype != context.dtype:
            raise ValueError(
                "UNet attn2 query and context must share device and dtype."
            )
