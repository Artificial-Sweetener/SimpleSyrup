# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute exact generic regional convolution LoRA rank-space deltas."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as functional

from ...domain.regional_activation_geometry import RegionalActivationLayout
from ...masking.regional_activation_mask_projection import RegionalActivationMaskBatch
from .convolution_execution_plan import (
    RegionalConvolutionExecutionPlan,
    RegionalConvolutionTargetUse,
)


class RegionalConvolutionExecutor:
    """Call one original convolution and add regional deltas in declared order."""

    def execute(
        self,
        original: Callable[..., object],
        inputs: torch.Tensor,
        *args: object,
        plan: RegionalConvolutionExecutionPlan,
        masks: RegionalActivationMaskBatch,
        schedule_strengths: tuple[float, ...],
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply each active mask between exact down/middle and up convolutions."""

        expected_spatial = self._validate_call(
            inputs,
            plan,
            masks,
            schedule_strengths,
        )
        original_output = original(inputs, *args, **kwargs)
        if not isinstance(original_output, torch.Tensor):
            raise TypeError(
                "Regional convolution original operation must return a tensor."
            )
        parameters = plan.uses[0].parameters
        expected_output = (
            int(inputs.shape[0]),
            parameters.output_channels,
            *expected_spatial,
        )
        if (
            tuple(original_output.shape) != expected_output
            or original_output.device != inputs.device
            or original_output.dtype != inputs.dtype
        ):
            raise ValueError(
                "Regional convolution original output must match the observed "
                "operation shape and input execution type."
            )
        result = original_output
        for index, use in enumerate(plan.uses):
            scale = use.base_strength * schedule_strengths[index]
            if scale == 0.0:
                continue
            multiplier = masks.multipliers[use.region_index]
            if not bool(torch.count_nonzero(multiplier)):
                continue
            result = result + self._delta(
                inputs,
                use=use,
                multiplier=multiplier,
                scale=scale,
            )
        return result

    @staticmethod
    def _delta(
        inputs: torch.Tensor,
        *,
        use: RegionalConvolutionTargetUse,
        multiplier: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Return one exact down/optional-middle/mask/up convolution delta."""

        parameters = use.parameters
        weights = use.preparation.weights(device=inputs.device, dtype=inputs.dtype)
        down_kernel = (
            (1,) * parameters.dimension
            if weights.middle is not None
            else parameters.kernel_size
        )
        down = _convolution_weight(
            weights.down,
            dimension=parameters.dimension,
            output_channels=int(weights.down.shape[0]),
            input_channels=parameters.input_channels // parameters.groups,
            kernel_size=down_kernel,
        )
        if parameters.groups > 1:
            down = down.repeat(parameters.groups, 1, *((1,) * parameters.dimension))
        rank_values = _convolution(
            inputs,
            down,
            stride=parameters.stride,
            padding=parameters.padding,
            dilation=parameters.dilation,
            groups=parameters.groups,
        )
        if weights.middle is not None:
            middle = weights.middle
            if parameters.groups > 1:
                middle = middle.repeat(
                    parameters.groups,
                    1,
                    *((1,) * parameters.dimension),
                )
            rank_values = _convolution(
                rank_values,
                middle,
                stride=(1,) * parameters.dimension,
                padding=(0,) * parameters.dimension,
                dilation=(1,) * parameters.dimension,
                groups=parameters.groups,
            )
        if tuple(multiplier.shape) != (
            int(rank_values.shape[0]),
            1,
            *tuple(int(value) for value in rank_values.shape[2:]),
        ):
            raise ValueError(
                "Regional convolution mask must match the exact rank activation."
            )
        rank_values = rank_values * multiplier * scale
        rank = int(weights.down.shape[0])
        up = _convolution_weight(
            weights.up,
            dimension=parameters.dimension,
            output_channels=parameters.output_channels,
            input_channels=rank,
            kernel_size=(1,) * parameters.dimension,
        )
        return _convolution(
            rank_values,
            up,
            stride=(1,) * parameters.dimension,
            padding=(0,) * parameters.dimension,
            dilation=(1,) * parameters.dimension,
            groups=parameters.groups,
        )

    @staticmethod
    def _validate_call(
        inputs: object,
        plan: object,
        masks: object,
        schedule_strengths: object,
    ) -> tuple[int, ...]:
        """Preflight invocation, output geometry, masks, and schedules."""

        if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
            raise TypeError("Regional convolution inputs must be a floating tensor.")
        if not isinstance(plan, RegionalConvolutionExecutionPlan):
            raise TypeError("Regional convolution execution requires a typed plan.")
        parameters = plan.uses[0].parameters
        if inputs.ndim != parameters.dimension + 2:
            raise ValueError("Regional convolution input dimensionality is invalid.")
        if int(inputs.shape[1]) != parameters.input_channels:
            raise ValueError("Regional convolution input channel count is invalid.")
        if not isinstance(masks, RegionalActivationMaskBatch):
            raise TypeError("Regional convolution execution requires activation masks.")
        expected_layout = {
            1: RegionalActivationLayout.DIRECT_CONVOLUTION_1D,
            2: RegionalActivationLayout.DIRECT_CONVOLUTION_2D,
            3: RegionalActivationLayout.DIRECT_CONVOLUTION_3D,
        }[parameters.dimension]
        if masks.geometry.layout is not expected_layout:
            raise ValueError(
                "Regional convolution mask geometry has the wrong dimension."
            )
        if (
            masks.multipliers.device != inputs.device
            or masks.multipliers.dtype != inputs.dtype
        ):
            raise ValueError(
                "Regional convolution masks must match input device and dtype."
            )
        if not isinstance(schedule_strengths, tuple) or len(schedule_strengths) != len(
            plan.uses
        ):
            raise ValueError("Regional convolution schedules must align to uses.")
        if any(
            isinstance(strength, bool)
            or not isinstance(strength, int | float)
            or not math.isfinite(float(strength))
            for strength in schedule_strengths
        ):
            raise TypeError("Regional convolution schedule strengths must be finite.")
        if any(
            use.region_index >= int(masks.multipliers.shape[0]) for use in plan.uses
        ):
            raise ValueError(
                "Regional convolution use references an unavailable region."
            )
        expected_spatial = _rank_spatial_shape(
            tuple(int(value) for value in inputs.shape[2:]),
            plan.uses[0],
        )
        expected_rank_channels = (
            int(plan.uses[0].preparation.down.shape[0]) * parameters.groups
        )
        if masks.geometry.invocation_shape != (
            int(inputs.shape[0]),
            expected_rank_channels,
            *expected_spatial,
        ):
            raise ValueError(
                "Regional convolution mask geometry must describe the exact rank "
                "activation shape."
            )
        multiplier_spatial = tuple(int(value) for value in masks.multipliers.shape[3:])
        if parameters.dimension == 1:
            multiplier_spatial = tuple(
                int(value) for value in masks.multipliers.shape[3:]
            )
        if tuple(masks.multipliers.shape[1:3]) != (int(inputs.shape[0]), 1):
            raise ValueError("Regional convolution mask batch is invalid.")
        if multiplier_spatial != expected_spatial:
            raise ValueError("Regional convolution mask spatial shape is invalid.")
        return expected_spatial


def _rank_spatial_shape(
    input_spatial: tuple[int, ...],
    use: RegionalConvolutionTargetUse,
) -> tuple[int, ...]:
    """Return exact rank-activation spatial dimensions before original execution."""

    parameters = use.parameters
    down = use.preparation.down
    if down.ndim == 2 or use.preparation.middle is None:
        down_kernel = parameters.kernel_size
    else:
        down_kernel = tuple(int(value) for value in down.shape[2:])
    spatial = tuple(
        _convolution_output_size(
            size,
            kernel,
            stride,
            padding,
            dilation,
        )
        for size, kernel, stride, padding, dilation in zip(
            input_spatial,
            down_kernel,
            parameters.stride,
            parameters.padding,
            parameters.dilation,
            strict=True,
        )
    )
    middle = use.preparation.middle
    if middle is not None:
        middle_kernel = tuple(int(value) for value in middle.shape[2:])
        spatial = tuple(
            _convolution_output_size(size, kernel, 1, 0, 1)
            for size, kernel in zip(spatial, middle_kernel, strict=True)
        )
    if any(size < 1 for size in spatial):
        raise ValueError("Regional convolution rank activation would be empty.")
    return spatial


def _convolution_output_size(
    size: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """Return Torch's ordinary convolution output size for one axis."""

    return ((size + 2 * padding - dilation * (kernel - 1) - 1) // stride) + 1


def _convolution_weight(
    weight: torch.Tensor,
    *,
    dimension: int,
    output_channels: int,
    input_channels: int,
    kernel_size: tuple[int, ...],
) -> torch.Tensor:
    """Return the exact convolution view of matrix or convolution weights."""

    expected = (output_channels, input_channels, *kernel_size)
    if weight.ndim == 2:
        return weight.reshape(expected)
    if tuple(weight.shape) != expected:
        raise ValueError("Regional convolution prepared weight shape is incompatible.")
    if weight.ndim != dimension + 2:
        raise ValueError("Regional convolution prepared weight rank is incompatible.")
    return weight


def _convolution(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    *,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
) -> torch.Tensor:
    """Dispatch one exact dimensional convolution without model-family policy."""

    function = {
        1: functional.conv1d,
        2: functional.conv2d,
        3: functional.conv3d,
    }[inputs.ndim - 2]
    return function(
        inputs,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


REGIONAL_CONVOLUTION_EXECUTOR = RegionalConvolutionExecutor()
