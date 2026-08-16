# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare and consume reusable compatible regional-LoRA rank projections."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .active_support import (
    REGIONAL_LORA_ACTIVE_SUPPORT_RESOLVER,
    RegionalLoraActiveSupportResolver,
)
from .fused_active_accumulation import (
    REGIONAL_LORA_FUSED_ACTIVE_ACCUMULATOR,
    RegionalLoraFusedActiveAccumulator,
)
from .ordered_accumulation import (
    ORDERED_TENSOR_ACCUMULATOR,
    OrderedTensorAccumulator,
)


@dataclass(frozen=True, slots=True)
class RegionalLoraCompatibleRankProjection:
    """Retain one shared A projection and its exact active-row contract."""

    leading_shape: tuple[int, ...]
    rank_values: torch.Tensor
    up: torch.Tensor
    multiplier_values: tuple[torch.Tensor, ...]
    indices: torch.Tensor | None

    def __post_init__(self) -> None:
        """Require aligned contiguous rank, B-weight, multiplier, and row values."""

        if (
            self.rank_values.ndim != 3
            or not self.rank_values.is_floating_point()
            or not self.rank_values.is_contiguous()
            or self.up.ndim != 3
            or self.up.device != self.rank_values.device
            or self.up.dtype != self.rank_values.dtype
            or not self.up.is_contiguous()
        ):
            raise ValueError("Compatible LoRA rank tensors are misaligned.")
        active_rows, target_count, rank = self.rank_values.shape
        if (
            target_count < 1
            or rank < 1
            or int(self.up.shape[0]) != target_count
            or int(self.up.shape[2]) != rank
        ):
            raise ValueError("Compatible LoRA rank dimensions are inconsistent.")
        if len(self.multiplier_values) != target_count or any(
            value.ndim != 1
            or int(value.shape[0]) != active_rows
            or value.device != self.rank_values.device
            or value.dtype != self.rank_values.dtype
            for value in self.multiplier_values
        ):
            raise ValueError("Compatible LoRA rank multipliers are misaligned.")
        output_rows = 1
        for size in self.leading_shape:
            if size < 1:
                raise ValueError("Compatible LoRA leading dimensions must be positive.")
            output_rows *= size
        if self.indices is None:
            if active_rows != output_rows:
                raise ValueError("Dense compatible LoRA projection requires every row.")
        elif (
            self.indices.ndim != 1
            or self.indices.dtype is not torch.int64
            or self.indices.device != self.rank_values.device
            or int(self.indices.shape[0]) != active_rows
        ):
            raise ValueError(
                "Sparse compatible LoRA projection indices are misaligned."
            )

    @property
    def target_count(self) -> int:
        """Return the number of compatible target projections."""

        return int(self.rank_values.shape[1])

    @property
    def output_features(self) -> int:
        """Return the common target output width."""

        return int(self.up.shape[1])


class RegionalLoraCompatibleRankProjector:
    """Own shared active A projection and exact B/output consumption."""

    def __init__(
        self,
        support_resolver: RegionalLoraActiveSupportResolver = (
            REGIONAL_LORA_ACTIVE_SUPPORT_RESOLVER
        ),
        accumulator: OrderedTensorAccumulator = ORDERED_TENSOR_ACCUMULATOR,
        fused_accumulator: RegionalLoraFusedActiveAccumulator = (
            REGIONAL_LORA_FUSED_ACTIVE_ACCUMULATOR
        ),
    ) -> None:
        """Retain support, ordered-addition, and fused-kernel authorities."""

        if not isinstance(support_resolver, RegionalLoraActiveSupportResolver):
            raise TypeError("Compatible LoRA projection requires a support resolver.")
        if not isinstance(accumulator, OrderedTensorAccumulator):
            raise TypeError("Compatible LoRA projection requires an accumulator.")
        if not isinstance(fused_accumulator, RegionalLoraFusedActiveAccumulator):
            raise TypeError("Compatible LoRA projection requires fused accumulation.")
        self._support_resolver = support_resolver
        self._accumulator = accumulator
        self._fused_accumulator = fused_accumulator

    def prepare(
        self,
        inputs: torch.Tensor,
        *,
        down: torch.Tensor,
        up: torch.Tensor,
        multipliers: tuple[torch.Tensor, ...],
        rank: int,
    ) -> RegionalLoraCompatibleRankProjection:
        """Project one exact union-support input batch through every compatible A."""

        self._validate_inputs(inputs, down, up, multipliers, rank)
        leading_shape = tuple(int(size) for size in inputs.shape[:-1])
        support = self._support_resolver.resolve(
            multipliers,
            leading_shape=leading_shape,
        )
        flattened_inputs = inputs.reshape(-1, int(inputs.shape[-1]))
        if support is None:
            active_inputs = flattened_inputs
            multiplier_values = tuple(
                multiplier.expand(leading_shape).reshape(-1)
                for multiplier in multipliers
            )
            indices = None
        else:
            active_inputs = flattened_inputs.index_select(0, support.indices)
            multiplier_values = support.multiplier_values
            indices = support.indices
        rank_values = functional.linear(active_inputs, down).reshape(
            int(active_inputs.shape[0]),
            len(multipliers),
            rank,
        )
        return RegionalLoraCompatibleRankProjection(
            leading_shape,
            rank_values,
            up,
            multiplier_values,
            indices,
        )

    def materialize_deltas(
        self,
        projection: RegionalLoraCompatibleRankProjection,
    ) -> tuple[torch.Tensor, ...]:
        """Materialize individually contiguous deltas for fallback consumers."""

        active = self._project_active(projection)
        if projection.indices is None:
            return tuple(
                active[target_index].reshape(
                    *projection.leading_shape,
                    projection.output_features,
                )
                for target_index in range(projection.target_count)
            )
        output_rows = 1
        for size in projection.leading_shape:
            output_rows *= size
        deltas = projection.rank_values.new_zeros(
            (projection.target_count, output_rows, projection.output_features)
        )
        if int(projection.indices.shape[0]) > 0:
            deltas.index_copy_(1, projection.indices, active)
        return tuple(
            deltas[target_index].reshape(
                *projection.leading_shape,
                projection.output_features,
            )
            for target_index in range(projection.target_count)
        )

    def add_all(
        self,
        original_output: torch.Tensor,
        projection: RegionalLoraCompatibleRankProjection,
    ) -> torch.Tensor:
        """Add every compatible target delta to one output in declared order."""

        flattened_output = self._validated_output(original_output, projection)
        if int(projection.rank_values.shape[0]) == 0:
            return original_output
        if self._fused_accumulator.supports(flattened_output, projection.up):
            self._fused_accumulator.add(
                flattened_output,
                rank_values=projection.rank_values,
                up=projection.up,
                multipliers=projection.multiplier_values,
                indices=projection.indices,
            )
            return flattened_output.reshape(original_output.shape)
        active_output = (
            flattened_output
            if projection.indices is None
            else flattened_output.index_select(0, projection.indices)
        )
        projected = self._project_active(projection)
        self._accumulator.accumulate(
            active_output,
            tuple(
                projected[target_index]
                for target_index in range(projection.target_count)
            ),
        )
        if projection.indices is not None:
            flattened_output.index_copy_(0, projection.indices, active_output)
        return flattened_output.reshape(original_output.shape)

    def add_target(
        self,
        original_output: torch.Tensor,
        projection: RegionalLoraCompatibleRankProjection,
        target_index: int,
    ) -> torch.Tensor:
        """Fuse one selected compatible B projection into its distinct output."""

        if (
            isinstance(target_index, bool)
            or not 0 <= target_index < projection.target_count
        ):
            raise IndexError("Compatible LoRA target index is out of range.")
        flattened_output = self._validated_output(original_output, projection)
        if int(projection.rank_values.shape[0]) == 0:
            return original_output
        if self._fused_accumulator.supports(flattened_output, projection.up):
            self._fused_accumulator.add_target(
                flattened_output,
                rank_values=projection.rank_values,
                up=projection.up,
                multiplier=projection.multiplier_values[target_index],
                indices=projection.indices,
                target_index=target_index,
            )
            return flattened_output.reshape(original_output.shape)
        delta = functional.linear(
            projection.rank_values[:, target_index]
            * projection.multiplier_values[target_index].unsqueeze(-1),
            projection.up[target_index],
        )
        if projection.indices is None:
            flattened_output.add_(delta)
        else:
            active_output = flattened_output.index_select(0, projection.indices)
            active_output.add_(delta)
            flattened_output.index_copy_(0, projection.indices, active_output)
        return flattened_output.reshape(original_output.shape)

    @staticmethod
    def _project_active(
        projection: RegionalLoraCompatibleRankProjection,
    ) -> torch.Tensor:
        """Project active rank rows into contiguous target-major deltas."""

        if int(projection.rank_values.shape[0]) == 0:
            return projection.rank_values.new_zeros(
                (projection.target_count, 0, projection.output_features)
            )
        weighted = projection.rank_values * torch.stack(
            projection.multiplier_values,
            dim=1,
        ).unsqueeze(-1)
        return torch.bmm(
            weighted.permute(1, 0, 2),
            projection.up.transpose(1, 2),
        )

    @staticmethod
    def _validated_output(
        original_output: torch.Tensor,
        projection: RegionalLoraCompatibleRankProjection,
    ) -> torch.Tensor:
        """Require one output aligned to the prepared execution contract."""

        expected_shape = (*projection.leading_shape, projection.output_features)
        if (
            not isinstance(original_output, torch.Tensor)
            or not original_output.is_floating_point()
            or tuple(original_output.shape) != expected_shape
            or original_output.device != projection.rank_values.device
            or original_output.dtype != projection.rank_values.dtype
        ):
            raise ValueError("Compatible LoRA output is misaligned.")
        return original_output.reshape(-1, projection.output_features)

    @staticmethod
    def _validate_inputs(
        inputs: torch.Tensor,
        down: torch.Tensor,
        up: torch.Tensor,
        multipliers: tuple[torch.Tensor, ...],
        rank: int,
    ) -> None:
        """Require one complete compatible A/B and multiplier contract."""

        if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
            raise TypeError("Compatible LoRA inputs must be a floating tensor.")
        target_count = len(multipliers)
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 1
            or target_count < 1
            or down.ndim != 2
            or up.ndim != 3
            or down.device != inputs.device
            or up.device != inputs.device
            or down.dtype != inputs.dtype
            or up.dtype != inputs.dtype
            or tuple(down.shape) != (target_count * rank, int(inputs.shape[-1]))
            or int(up.shape[0]) != target_count
            or int(up.shape[2]) != rank
        ):
            raise ValueError("Compatible LoRA projection tensors are misaligned.")


REGIONAL_LORA_COMPATIBLE_RANK_PROJECTOR = RegionalLoraCompatibleRankProjector()
