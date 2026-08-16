# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute compatible unique-target regional Linear projections on CUDA."""

from __future__ import annotations

import torch
from torch.nn import functional

from .active_support import (
    REGIONAL_LORA_ACTIVE_SUPPORT_RESOLVER,
    RegionalLoraActiveSupportResolver,
)
from .fused_active_accumulation import (
    REGIONAL_LORA_FUSED_ACTIVE_ACCUMULATOR,
    RegionalLoraFusedActiveAccumulator,
)
from .linear_mapped_projection_plan import RegionalLinearMappedProjectionPlan


class RegionalMappedLinearProjector:
    """Own unique A projection and mapped ordered fused-B accumulation."""

    def __init__(
        self,
        support_resolver: RegionalLoraActiveSupportResolver = (
            REGIONAL_LORA_ACTIVE_SUPPORT_RESOLVER
        ),
        accumulator: RegionalLoraFusedActiveAccumulator = (
            REGIONAL_LORA_FUSED_ACTIVE_ACCUMULATOR
        ),
    ) -> None:
        """Retain focused support and fused accumulation collaborators."""

        if not isinstance(support_resolver, RegionalLoraActiveSupportResolver):
            raise TypeError("Mapped Linear support resolver is invalid.")
        if not isinstance(accumulator, RegionalLoraFusedActiveAccumulator):
            raise TypeError("Mapped Linear fused accumulator is invalid.")
        self._support_resolver = support_resolver
        self._accumulator = accumulator

    def add(
        self,
        output: torch.Tensor,
        inputs: torch.Tensor,
        *,
        plan: RegionalLinearMappedProjectionPlan,
        multipliers: tuple[torch.Tensor | None, ...],
        leading_shape: tuple[int, ...],
    ) -> bool:
        """Add all declared groups or decline an incompatible execution call."""

        if (
            not plan.supported
            or inputs.device.type != "cuda"
            or inputs.dtype not in (torch.bfloat16, torch.float16)
            or len(multipliers) != len(plan.group_target_indices)
            or any(multiplier is None for multiplier in multipliers)
        ):
            return False
        active_multipliers = tuple(
            multiplier for multiplier in multipliers if multiplier is not None
        )
        if len(active_multipliers) != len(multipliers):
            raise AssertionError("Mapped Linear multiplier disappeared.")
        support = self._support_resolver.resolve(
            active_multipliers,
            leading_shape=leading_shape,
        )
        if support is None:
            active_inputs = inputs
            multiplier_values = tuple(
                multiplier.expand(leading_shape).reshape(-1)
                for multiplier in active_multipliers
            )
            indices = None
        else:
            indices = support.indices
            if int(indices.shape[0]) == 0:
                return True
            active_inputs = inputs.index_select(0, indices)
            multiplier_values = support.multiplier_values
        weights = plan.weights(device=inputs.device, dtype=inputs.dtype)
        active_rows = int(active_inputs.shape[0])
        rank_values = functional.linear(active_inputs, weights.down).reshape(
            active_rows,
            plan.target_count,
            plan.rank,
        )
        self._accumulator.add_mapped(
            output,
            rank_values=rank_values,
            up=weights.up,
            multipliers=multiplier_values,
            indices=indices,
            target_indices=plan.target_indices(inputs.device),
        )
        return True


REGIONAL_MAPPED_LINEAR_PROJECTOR = RegionalMappedLinearProjector()
