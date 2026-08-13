# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt bound Comfy convolution LoRAs to exact generic execution plans."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from comfy.weight_adapter.lora import LoRAAdapter
from torch import nn

from .convolution_preparation import RegionalConvolutionPreparation
from .target_binding import (
    BoundRegionalLoraModuleClass,
    BoundRegionalLoraOperation,
    BoundRegionalLoraSpatialCapability,
)


@dataclass(frozen=True, slots=True)
class RegionalConvolutionParameters:
    """Retain the observed original operation parameters used by the down path."""

    dimension: int
    stride: tuple[int, ...]
    padding: tuple[int, ...]
    dilation: tuple[int, ...]
    groups: int
    input_channels: int
    output_channels: int
    kernel_size: tuple[int, ...]

    def __post_init__(self) -> None:
        """Require complete positive convolution metadata for one dimension."""

        if self.dimension not in (1, 2, 3):
            raise ValueError("Regional convolution dimension must be 1, 2, or 3.")
        for name, values, allow_zero in (
            ("stride", self.stride, False),
            ("padding", self.padding, True),
            ("dilation", self.dilation, False),
        ):
            if len(values) != self.dimension or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if allow_zero else 1)
                for value in values
            ):
                raise ValueError(f"Regional convolution {name} is invalid.")
        if (
            isinstance(self.groups, bool)
            or not isinstance(self.groups, int)
            or self.groups < 1
        ):
            raise ValueError("Regional convolution groups must be positive.")
        for name, value in (
            ("input_channels", self.input_channels),
            ("output_channels", self.output_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Regional convolution {name} must be positive.")
        if len(self.kernel_size) != self.dimension or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.kernel_size
        ):
            raise ValueError("Regional convolution kernel size is invalid.")
        if self.input_channels % self.groups or self.output_channels % self.groups:
            raise ValueError("Regional convolution channels must divide over groups.")


@dataclass(frozen=True, slots=True)
class RegionalConvolutionTargetUse:
    """Retain one ordered convolution adapter use and its exact tensor owner."""

    composition_index: int
    region_index: int
    operation_identity: tuple[int, int, int | None]
    preparation: RegionalConvolutionPreparation
    parameters: RegionalConvolutionParameters
    base_strength: float

    def __post_init__(self) -> None:
        """Require canonical indices, identities, owners, and finite strength."""

        for name, value in (
            ("composition_index", self.composition_index),
            ("region_index", self.region_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Regional convolution {name} must be a non-negative integer."
                )
        if not isinstance(self.preparation, RegionalConvolutionPreparation):
            raise TypeError("Regional convolution use requires preparation.")
        if not isinstance(self.parameters, RegionalConvolutionParameters):
            raise TypeError("Regional convolution use requires operation parameters.")
        if not isinstance(self.base_strength, float) or not math.isfinite(
            self.base_strength
        ):
            raise TypeError("Regional convolution base strength must be finite.")


@dataclass(slots=True)
class RegionalConvolutionExecutionPlan:
    """Retain one target module's declared-order convolution adapter uses."""

    uses: tuple[RegionalConvolutionTargetUse, ...]

    def __post_init__(self) -> None:
        """Require one ordered set sharing exact operation parameters and channels."""

        if not isinstance(self.uses, tuple) or not self.uses:
            raise ValueError("Regional convolution execution plan cannot be empty.")
        if any(not isinstance(use, RegionalConvolutionTargetUse) for use in self.uses):
            raise TypeError("Regional convolution plan contains an invalid use.")
        indices = tuple(use.composition_index for use in self.uses)
        if indices != tuple(sorted(indices)):
            raise ValueError("Regional convolution uses must follow composition order.")
        parameters = self.uses[0].parameters
        if any(use.parameters != parameters for use in self.uses):
            raise ValueError(
                "Regional convolution uses must share operation parameters."
            )

    def clear(self) -> None:
        """Release every locally retained prepared convolution tensor set."""

        for use in self.uses:
            use.preparation.clear()


class RegionalConvolutionExecutionPlanFactory:
    """Build one exact convolution plan from U2/U3/U4 binding evidence."""

    def build(
        self,
        bindings: tuple[BoundRegionalLoraOperation, ...],
    ) -> RegionalConvolutionExecutionPlan:
        """Adapt exact tensors and observed module parameters without mutation."""

        if not isinstance(bindings, tuple) or not bindings:
            raise ValueError("Regional convolution plan requires bound operations.")
        return RegionalConvolutionExecutionPlan(
            tuple(self._use(item) for item in bindings)
        )

    @staticmethod
    def _use(binding: BoundRegionalLoraOperation) -> RegionalConvolutionTargetUse:
        """Adapt one direct bound convolution operation."""

        expected_classes = {
            BoundRegionalLoraModuleClass.CONVOLUTION_1D: (nn.Conv1d, 1),
            BoundRegionalLoraModuleClass.CONVOLUTION_2D: (nn.Conv2d, 2),
            BoundRegionalLoraModuleClass.CONVOLUTION_3D: (nn.Conv3d, 3),
        }
        if not isinstance(binding, BoundRegionalLoraOperation) or not binding.bound:
            raise ValueError(
                "Regional convolution execution requires a bound operation."
            )
        if binding.module_class not in expected_classes:
            raise ValueError(
                "Regional convolution execution requires a convolution target."
            )
        if binding.spatial_capability is not BoundRegionalLoraSpatialCapability.DIRECT:
            raise ValueError(
                "Regional convolution execution requires direct capability."
            )
        module_type, dimension = expected_classes[binding.module_class]
        if not isinstance(binding.module, module_type):
            raise TypeError("Regional convolution bound module class is inconsistent.")
        operation = binding.normalized_target.operation
        if not isinstance(operation, LoRAAdapter):
            raise TypeError("Regional convolution operation must be LoRAAdapter.")
        up, down, _alpha, middle, dora, reshape = operation.weights
        if not isinstance(up, torch.Tensor) or not isinstance(down, torch.Tensor):
            raise TypeError("Regional convolution down/up weights must be tensors.")
        if middle is not None and not isinstance(middle, torch.Tensor):
            raise TypeError("Regional convolution middle weight must be a tensor.")
        if dora is not None or reshape is not None:
            raise ValueError(
                "Regional convolution ordinary execution rejects DoRA/reshape."
            )
        descriptor = binding.descriptor
        if descriptor.intrinsic_scale is None:
            raise ValueError("Regional convolution descriptor lacks intrinsic scale.")
        module = binding.module
        assert isinstance(module, nn.Conv1d | nn.Conv2d | nn.Conv3d)
        if module.padding_mode != "zeros":
            raise ValueError(
                "Regional convolution execution requires zero padding mode."
            )
        parameters = RegionalConvolutionParameters(
            dimension,
            tuple(int(value) for value in module.stride),
            tuple(int(value) for value in module.padding),
            tuple(int(value) for value in module.dilation),
            module.groups,
            module.in_channels,
            module.out_channels,
            tuple(int(value) for value in module.kernel_size),
        )
        middle_tensor = middle if isinstance(middle, torch.Tensor) else None
        _validate_adapter_shapes(
            down,
            middle_tensor,
            up,
            parameters=parameters,
        )
        return RegionalConvolutionTargetUse(
            descriptor.adapter.composition_index,
            descriptor.adapter.region_index,
            (id(down), id(up), None if middle_tensor is None else id(middle_tensor)),
            RegionalConvolutionPreparation(down, middle_tensor, up),
            parameters,
            float(descriptor.adapter.model_strength * descriptor.intrinsic_scale),
        )


REGIONAL_CONVOLUTION_EXECUTION_PLAN_FACTORY = RegionalConvolutionExecutionPlanFactory()


def _validate_adapter_shapes(
    down: torch.Tensor,
    middle: torch.Tensor | None,
    up: torch.Tensor,
    *,
    parameters: RegionalConvolutionParameters,
) -> None:
    """Require exact matrix/conv down, optional middle, and pointwise up shapes."""

    rank = int(down.shape[0])
    input_per_group = parameters.input_channels // parameters.groups
    expected_matrix_input = input_per_group * math.prod(parameters.kernel_size)
    if down.ndim == 2:
        if tuple(down.shape) != (rank, expected_matrix_input):
            raise ValueError("Regional convolution matrix down shape is incompatible.")
        down_kernel = parameters.kernel_size
    elif down.ndim == parameters.dimension + 2:
        if int(down.shape[1]) != input_per_group:
            raise ValueError("Regional convolution down channels are incompatible.")
        down_kernel = tuple(int(value) for value in down.shape[2:])
    else:
        raise ValueError("Regional convolution down dimensionality is incompatible.")
    if middle is None:
        if down_kernel != parameters.kernel_size:
            raise ValueError("Regional convolution down kernel must match the module.")
    else:
        if down_kernel != (1,) * parameters.dimension:
            raise ValueError("LoCon down convolution must use a pointwise kernel.")
        if tuple(middle.shape) != (
            rank,
            rank,
            *parameters.kernel_size,
        ):
            raise ValueError("LoCon middle convolution shape is incompatible.")
    pointwise = (1,) * parameters.dimension
    if up.ndim == 2:
        valid_up = tuple(up.shape) == (parameters.output_channels, rank)
    else:
        valid_up = tuple(up.shape) == (
            parameters.output_channels,
            rank,
            *pointwise,
        )
    if not valid_up:
        raise ValueError("Regional convolution up shape must be pointwise.")
