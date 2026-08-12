# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute exact standard regional-LoRA low-rank deltas."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

from .compatible_rank_projection import (
    REGIONAL_LORA_COMPATIBLE_RANK_PROJECTOR,
    RegionalLoraCompatibleRankProjection,
    RegionalLoraCompatibleRankProjector,
)
from .ordered_accumulation import (
    ORDERED_TENSOR_ACCUMULATOR,
    OrderedTensorAccumulator,
)
from .preparation import (
    RegionalLoraCompatibleBatchPreparation,
    RegionalLoraTargetPreparation,
)
from .single_active_projection import (
    REGIONAL_LORA_SINGLE_ACTIVE_PROJECTION_EXECUTOR,
    RegionalLoraSingleActiveProjectionExecutor,
)


class RegionalLoraDeltaExecutor:
    """Compute exact unmerged full-rank standard LoRA deltas."""

    def __init__(
        self,
        single_projection: RegionalLoraSingleActiveProjectionExecutor = (
            REGIONAL_LORA_SINGLE_ACTIVE_PROJECTION_EXECUTOR
        ),
        compatible_projection: RegionalLoraCompatibleRankProjector = (
            REGIONAL_LORA_COMPATIBLE_RANK_PROJECTOR
        ),
        accumulator: OrderedTensorAccumulator = ORDERED_TENSOR_ACCUMULATOR,
    ) -> None:
        """Retain exact active-support and ordered-accumulation owners."""

        if not isinstance(
            single_projection,
            RegionalLoraSingleActiveProjectionExecutor,
        ):
            raise TypeError("Regional LoRA delta execution requires single projection.")
        if not isinstance(
            compatible_projection,
            RegionalLoraCompatibleRankProjector,
        ):
            raise TypeError(
                "Regional LoRA delta execution requires compatible projection."
            )
        if not isinstance(accumulator, OrderedTensorAccumulator):
            raise TypeError("Regional LoRA delta execution requires accumulation.")
        self._single_projection = single_projection
        self._compatible_projection = compatible_projection
        self._accumulator = accumulator

    def delta(
        self,
        inputs: torch.Tensor,
        *,
        preparation: RegionalLoraTargetPreparation,
        strength: float,
    ) -> torch.Tensor:
        """Return `strength * ((x @ A.T) @ B.T)` in the input execution type."""

        if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
            raise TypeError("Regional LoRA inputs must be a floating tensor.")
        if (
            inputs.ndim < 1
            or int(inputs.shape[-1]) != preparation.target.input_features
        ):
            raise ValueError(
                "Regional LoRA input feature dimension does not match its target."
            )
        if (
            isinstance(strength, bool)
            or not isinstance(strength, int | float)
            or not math.isfinite(float(strength))
        ):
            raise TypeError("Regional LoRA strength must be finite numeric data.")
        weights = preparation.weights(device=inputs.device, dtype=inputs.dtype)
        rank_values = functional.linear(inputs, weights.down)
        return functional.linear(rank_values, weights.up) * float(strength)

    def compatible_batch(
        self,
        inputs: torch.Tensor,
        *,
        preparation: RegionalLoraCompatibleBatchPreparation,
        multipliers: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Return masked full-rank group deltas with masks applied before B."""

        projection = self.prepare_compatible_rank_projection(
            inputs,
            preparation=preparation,
            multipliers=multipliers,
        )
        return self._compatible_projection.deltas(projection)

    def prepare_compatible_rank_projection(
        self,
        inputs: torch.Tensor,
        *,
        preparation: RegionalLoraCompatibleBatchPreparation,
        multipliers: tuple[torch.Tensor, ...],
    ) -> RegionalLoraCompatibleRankProjection:
        """Prepare one reusable compatible A projection over exact active rows."""

        if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
            raise TypeError("Regional LoRA inputs must be a floating tensor.")
        if inputs.ndim < 1 or int(inputs.shape[-1]) != preparation.input_features:
            raise ValueError(
                "Regional LoRA input feature dimension does not match its batch."
            )
        if len(multipliers) != len(preparation.targets):
            raise ValueError(
                "Regional LoRA multiplier count does not match its compatible batch."
            )
        leading_shape = tuple(int(size) for size in inputs.shape[:-1])
        for multiplier in multipliers:
            if (
                not isinstance(multiplier, torch.Tensor)
                or multiplier.device != inputs.device
                or multiplier.dtype != inputs.dtype
                or multiplier.ndim != len(leading_shape)
                or any(
                    observed not in (1, expected)
                    for observed, expected in zip(
                        multiplier.shape,
                        leading_shape,
                        strict=True,
                    )
                )
            ):
                raise ValueError(
                    "Regional LoRA multiplier cannot broadcast to its input."
                )
        weights = preparation.weights(device=inputs.device, dtype=inputs.dtype)
        return self._compatible_projection.prepare(
            inputs,
            down=weights.down,
            up=weights.up,
            multipliers=multipliers,
            rank=preparation.rank,
        )

    def masked_delta(
        self,
        inputs: torch.Tensor,
        *,
        preparation: RegionalLoraTargetPreparation,
        multiplier: torch.Tensor,
    ) -> torch.Tensor:
        """Return one rank-space-masked delta through two direct linear kernels."""

        if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
            raise TypeError("Regional LoRA inputs must be a floating tensor.")
        if (
            inputs.ndim < 1
            or int(inputs.shape[-1]) != preparation.target.input_features
        ):
            raise ValueError(
                "Regional LoRA input feature dimension does not match its target."
            )
        leading_shape = tuple(int(size) for size in inputs.shape[:-1])
        if (
            not isinstance(multiplier, torch.Tensor)
            or multiplier.device != inputs.device
            or multiplier.dtype != inputs.dtype
            or multiplier.ndim != len(leading_shape)
            or any(
                observed not in (1, expected)
                for observed, expected in zip(
                    multiplier.shape,
                    leading_shape,
                    strict=True,
                )
            )
        ):
            observed = (
                tuple(multiplier.shape)
                if isinstance(multiplier, torch.Tensor)
                else type(multiplier).__name__
            )
            raise ValueError(
                "Regional LoRA multiplier cannot broadcast to its input; "
                f"observed {observed!r}, device "
                f"{getattr(multiplier, 'device', None)!s}, dtype "
                f"{getattr(multiplier, 'dtype', None)!s} for leading shape "
                f"{leading_shape!r}, device {inputs.device!s}, dtype {inputs.dtype!s}."
            )
        weights = preparation.weights(device=inputs.device, dtype=inputs.dtype)
        return self._single_projection.delta(
            inputs,
            down=weights.down,
            up=weights.up,
            multiplier=multiplier,
        )

    def add_masked_delta(
        self,
        original_output: torch.Tensor,
        inputs: torch.Tensor,
        *,
        preparation: RegionalLoraTargetPreparation,
        multiplier: torch.Tensor,
    ) -> torch.Tensor:
        """Accumulate one masked full-rank delta in its B projection kernel."""

        if (
            not isinstance(original_output, torch.Tensor)
            or not original_output.is_floating_point()
        ):
            raise TypeError("Regional LoRA original output must be a floating tensor.")
        if (
            original_output.device != inputs.device
            or original_output.dtype != inputs.dtype
        ):
            raise ValueError(
                "Regional LoRA original output must match its input execution type."
            )
        expected_shape = (
            *inputs.shape[:-1],
            preparation.target.output_features,
        )
        if tuple(original_output.shape) != expected_shape:
            raise ValueError(
                "Regional LoRA original output does not match its target shape."
            )
        leading_shape = tuple(int(size) for size in inputs.shape[:-1])
        if (
            not isinstance(multiplier, torch.Tensor)
            or multiplier.device != inputs.device
            or multiplier.dtype != inputs.dtype
            or multiplier.ndim != len(leading_shape)
            or any(
                observed not in (1, expected)
                for observed, expected in zip(
                    multiplier.shape,
                    leading_shape,
                    strict=True,
                )
            )
        ):
            observed = (
                tuple(multiplier.shape)
                if isinstance(multiplier, torch.Tensor)
                else type(multiplier).__name__
            )
            raise ValueError(
                "Regional LoRA multiplier cannot broadcast to its input; "
                f"observed {observed!r}, device "
                f"{getattr(multiplier, 'device', None)!s}, dtype "
                f"{getattr(multiplier, 'dtype', None)!s} for leading shape "
                f"{leading_shape!r}, device {inputs.device!s}, dtype {inputs.dtype!s}."
            )
        weights = preparation.weights(device=inputs.device, dtype=inputs.dtype)
        return self._single_projection.add(
            original_output,
            inputs,
            down=weights.down,
            up=weights.up,
            multiplier=multiplier,
        )

    def add_compatible_deltas(
        self,
        original_output: torch.Tensor,
        inputs: torch.Tensor,
        *,
        preparation: RegionalLoraCompatibleBatchPreparation,
        multipliers: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Batch compatible deltas and accumulate only their exact union support."""

        projection = self.prepare_compatible_rank_projection(
            inputs,
            preparation=preparation,
            multipliers=multipliers,
        )
        return self._compatible_projection.add_all(original_output, projection)

    def add_preprojected_target(
        self,
        original_output: torch.Tensor,
        projection: RegionalLoraCompatibleRankProjection,
        target_index: int,
    ) -> torch.Tensor:
        """Add one target B projection from an already shared A projection."""

        return self._compatible_projection.add_target(
            original_output,
            projection,
            target_index,
        )

    def accumulate_ordered(
        self,
        original_output: torch.Tensor,
        deltas: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Delegate exact declared-order output accumulation to its owner."""

        return self._accumulator.accumulate(original_output, deltas)


REGIONAL_LORA_DELTA_EXECUTOR = RegionalLoraDeltaExecutor()
