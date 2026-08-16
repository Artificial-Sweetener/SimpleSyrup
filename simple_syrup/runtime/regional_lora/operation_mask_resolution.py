# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compose projected spatial masks with target-use CFG branch ownership."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch

from ...domain.regional_activation_geometry import RegionalActivationGeometry
from ...domain.regional_attention import RegionalAttentionBranch
from ...domain.regional_attention_batch import BatchedRegionalAttentionContexts
from ...domain.regional_lora_plan import RegionalLoraBranch
from ...masking.regional_activation_mask_projection import (
    RegionalActivationMaskBatch,
)
from .linear_invocation_preparation import RegionalLinearInvocationPreparationCache


class RegionalOperationMaskUse(Protocol):
    """Expose immutable ownership fields shared by Linear and Conv target uses."""

    @property
    def composition_index(self) -> int:
        """Return the global adapter composition index."""

        ...

    @property
    def region_index(self) -> int:
        """Return the authored region index."""

        ...

    @property
    def branch(self) -> RegionalLoraBranch:
        """Return the authored positive or negative branch."""

        ...


@dataclass(frozen=True, slots=True)
class RegionalOperationMaskBatch:
    """Retain one activation-shaped multiplier for every ordered target use."""

    multipliers: torch.Tensor
    geometry: RegionalActivationGeometry
    composition_indices: tuple[int, ...]
    linear_invocations: RegionalLinearInvocationPreparationCache = field(
        default_factory=RegionalLinearInvocationPreparationCache,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Require exact use order and a valid activation-mask tensor contract."""

        spatial = RegionalActivationMaskBatch(self.multipliers, self.geometry)
        if not isinstance(self.composition_indices, tuple) or not (
            self.composition_indices
        ):
            raise ValueError("Regional operation masks require composition indices.")
        if len(self.composition_indices) != int(spatial.multipliers.shape[0]):
            raise ValueError(
                "Regional operation mask count must match composition indices."
            )
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.composition_indices
        ):
            raise ValueError(
                "Regional operation composition indices must be non-negative integers."
            )
        if self.composition_indices != tuple(sorted(self.composition_indices)):
            raise ValueError(
                "Regional operation masks must follow global composition order."
            )


class RegionalOperationMaskResolver:
    """Apply existing aligned chunk ownership to projected regional masks."""

    def resolve(
        self,
        spatial_masks: RegionalActivationMaskBatch,
        *,
        contexts: BatchedRegionalAttentionContexts,
        uses: Sequence[RegionalOperationMaskUse],
    ) -> RegionalOperationMaskBatch:
        """Return use-indexed masks with non-owning CFG rows set to zero."""

        if not isinstance(spatial_masks, RegionalActivationMaskBatch):
            raise TypeError("Regional operation resolution requires spatial masks.")
        if not isinstance(contexts, BatchedRegionalAttentionContexts):
            raise TypeError("Regional operation resolution requires contexts.")
        if not isinstance(uses, Sequence) or not uses:
            raise ValueError("Regional operation resolution requires target uses.")
        invocation_batch = spatial_masks.geometry.batch_alignment.invocation_batch_size
        if int(contexts.base_context.shape[0]) != invocation_batch:
            raise ValueError(
                "Regional operation contexts must match the activation invocation "
                "batch."
            )
        composition_indices = tuple(use.composition_index for use in uses)
        if composition_indices != tuple(sorted(composition_indices)):
            raise ValueError(
                "Regional operation uses must follow global composition order."
            )
        use_masks = tuple(
            self._use_mask(spatial_masks, contexts=contexts, use=use) for use in uses
        )
        return RegionalOperationMaskBatch(
            torch.stack(use_masks),
            spatial_masks.geometry,
            composition_indices,
        )

    @staticmethod
    def _use_mask(
        spatial_masks: RegionalActivationMaskBatch,
        *,
        contexts: BatchedRegionalAttentionContexts,
        use: RegionalOperationMaskUse,
    ) -> torch.Tensor:
        """Return one spatial mask gated to the use's exact CFG branch."""

        if (
            isinstance(use.composition_index, bool)
            or not isinstance(use.composition_index, int)
            or use.composition_index < 0
        ):
            raise ValueError(
                "Regional operation composition index must be non-negative."
            )
        if (
            isinstance(use.region_index, bool)
            or not isinstance(use.region_index, int)
            or use.region_index < 0
        ):
            raise ValueError("Regional operation region index must be non-negative.")
        if use.region_index >= int(spatial_masks.multipliers.shape[0]):
            raise ValueError("Regional operation use references an unavailable region.")
        if not isinstance(use.branch, RegionalLoraBranch):
            raise TypeError("Regional operation use branch has an invalid type.")
        gate = REGIONAL_OPERATION_BRANCH_GATE_RESOLVER.resolve(
            contexts,
            branch=use.branch,
            authority=spatial_masks.multipliers,
        )
        gate_shape = (int(gate.shape[0]),) + (1,) * (spatial_masks.multipliers.ndim - 2)
        return spatial_masks.multipliers[use.region_index] * gate.reshape(gate_shape)


class RegionalOperationBranchGateResolver:
    """Resolve authored LoRA branch ownership from aligned context chunks."""

    def resolve(
        self,
        contexts: BatchedRegionalAttentionContexts,
        *,
        branch: RegionalLoraBranch,
        authority: torch.Tensor,
    ) -> torch.Tensor:
        """Return one source-batch scalar gate on the authority execution type."""

        if not isinstance(contexts, BatchedRegionalAttentionContexts):
            raise TypeError("Regional branch gating requires aligned contexts.")
        if not isinstance(branch, RegionalLoraBranch):
            raise TypeError("Regional branch gating requires a LoRA branch.")
        if not isinstance(authority, torch.Tensor) or not authority.is_floating_point():
            raise TypeError("Regional branch gating requires a floating authority.")
        expected_branch = (
            RegionalAttentionBranch.POSITIVE
            if branch is RegionalLoraBranch.POSITIVE
            else RegionalAttentionBranch.NEGATIVE
        )
        gate = authority.new_zeros((int(contexts.base_context.shape[0]),))
        for chunk in contexts.chunks:
            if chunk.branch is expected_branch:
                gate[chunk.batch_start : chunk.batch_stop] = 1.0
        return gate


REGIONAL_OPERATION_MASK_RESOLVER = RegionalOperationMaskResolver()
REGIONAL_OPERATION_BRANCH_GATE_RESOLVER = RegionalOperationBranchGateResolver()
