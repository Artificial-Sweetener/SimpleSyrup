# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute one regional-LoRA target only on its exact nonzero support."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from .active_support import (
    REGIONAL_LORA_ACTIVE_SUPPORT_RESOLVER,
    RegionalLoraActiveSupport,
    RegionalLoraActiveSupportResolver,
)
from .fused_active_accumulation import (
    REGIONAL_LORA_FUSED_ACTIVE_ACCUMULATOR,
    RegionalLoraFusedActiveAccumulator,
)


class RegionalLoraSingleActiveProjectionExecutor:
    """Own one target's gather, low-rank projection, and source-order scatter."""

    def __init__(
        self,
        support_resolver: RegionalLoraActiveSupportResolver = (
            REGIONAL_LORA_ACTIVE_SUPPORT_RESOLVER
        ),
        fused_accumulator: RegionalLoraFusedActiveAccumulator = (
            REGIONAL_LORA_FUSED_ACTIVE_ACCUMULATOR
        ),
    ) -> None:
        """Retain exact support and fused output-accumulation authorities."""

        if not isinstance(support_resolver, RegionalLoraActiveSupportResolver):
            raise TypeError("Single LoRA projection requires a support resolver.")
        if not isinstance(fused_accumulator, RegionalLoraFusedActiveAccumulator):
            raise TypeError("Single LoRA projection requires fused accumulation.")
        self._support_resolver = support_resolver
        self._fused_accumulator = fused_accumulator

    def delta(
        self,
        inputs: torch.Tensor,
        *,
        down: torch.Tensor,
        up: torch.Tensor,
        multiplier: torch.Tensor,
    ) -> torch.Tensor:
        """Return one full-shaped delta while projecting only nonzero positions."""

        leading_shape = tuple(int(size) for size in inputs.shape[:-1])
        support = self._support_resolver.resolve(
            (multiplier,),
            leading_shape=leading_shape,
        )
        if support is None:
            rank_values = functional.linear(inputs, down)
            return functional.linear(rank_values * multiplier.unsqueeze(-1), up)
        output_features = int(up.shape[0])
        flattened = inputs.reshape(-1, int(inputs.shape[-1]))
        delta = inputs.new_zeros((int(flattened.shape[0]), output_features))
        active = self._active_delta(
            flattened,
            support=support,
            down=down,
            up=up,
        )
        if int(support.indices.shape[0]) > 0:
            delta.index_copy_(0, support.indices, active)
        return delta.reshape(*leading_shape, output_features)

    def add(
        self,
        original_output: torch.Tensor,
        inputs: torch.Tensor,
        *,
        down: torch.Tensor,
        up: torch.Tensor,
        multiplier: torch.Tensor,
    ) -> torch.Tensor:
        """Add one target directly at its exact nonzero output rows."""

        leading_shape = tuple(int(size) for size in inputs.shape[:-1])
        support = self._support_resolver.resolve(
            (multiplier,),
            leading_shape=leading_shape,
        )
        flattened_inputs = inputs.reshape(-1, int(inputs.shape[-1]))
        flattened_output = original_output.reshape(-1, int(original_output.shape[-1]))
        up_batch = up.unsqueeze(0)
        if self._fused_accumulator.supports(flattened_output, up_batch):
            multiplier_values: tuple[torch.Tensor, ...]
            indices: torch.Tensor | None
            if support is None:
                active_inputs = flattened_inputs
                multiplier_values = (multiplier.expand(leading_shape).reshape(-1),)
                indices = None
            else:
                if int(support.indices.shape[0]) == 0:
                    return original_output
                active_inputs = flattened_inputs.index_select(0, support.indices)
                multiplier_values = support.multiplier_values
                indices = support.indices
            rank_values = functional.linear(active_inputs, down).reshape(
                int(active_inputs.shape[0]),
                1,
                int(down.shape[0]),
            )
            self._fused_accumulator.add(
                flattened_output,
                rank_values=rank_values,
                up=up_batch,
                multipliers=multiplier_values,
                indices=indices,
            )
            return flattened_output.reshape(original_output.shape)
        if support is None:
            rank_values = functional.linear(inputs, down)
            rank_values.mul_(multiplier.unsqueeze(-1))
            flattened_output.addmm_(
                rank_values.reshape(-1, int(down.shape[0])),
                up.transpose(0, 1),
            )
            return flattened_output.reshape(original_output.shape)
        if int(support.indices.shape[0]) == 0:
            return original_output
        active_output = flattened_output.index_select(0, support.indices)
        active_output += self._active_delta(
            flattened_inputs,
            support=support,
            down=down,
            up=up,
        )
        flattened_output.index_copy_(0, support.indices, active_output)
        return flattened_output.reshape(original_output.shape)

    @staticmethod
    def _active_delta(
        flattened_inputs: torch.Tensor,
        *,
        support: RegionalLoraActiveSupport,
        down: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        """Project one gathered input and exact gathered multiplier vector."""

        if int(support.indices.shape[0]) == 0:
            return flattened_inputs.new_zeros((0, int(up.shape[0])))
        active_inputs = flattened_inputs.index_select(0, support.indices)
        rank_values = functional.linear(active_inputs, down)
        return functional.linear(
            rank_values * support.multiplier_values[0].unsqueeze(-1),
            up,
        )


REGIONAL_LORA_SINGLE_ACTIVE_PROJECTION_EXECUTOR = (
    RegionalLoraSingleActiveProjectionExecutor()
)
