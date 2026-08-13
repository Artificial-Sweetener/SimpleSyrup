# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate resolved ordinary-LoRA shapes against observed Torch operations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from ...domain.resolved_regional_lora import (
    ResolvedLoraOperationClass,
    ResolvedRegionalLoraOperation,
)
from .target_binding import BoundRegionalLoraModuleClass


@dataclass(frozen=True, slots=True)
class TargetShapeAdmission:
    """Retain one observed module class, parameter, and shape failure if any."""

    module_class: BoundRegionalLoraModuleClass
    parameter: object | None
    issue: str | None


class RegionalLoraTargetShapeValidator:
    """Validate target modules and parameter shapes without tensor movement."""

    def validate(
        self,
        descriptor: ResolvedRegionalLoraOperation,
        module: object,
        parameter: object,
    ) -> TargetShapeAdmission:
        """Return complete shape admission for one observed target module."""

        module_class = _module_class(module)
        if module_class is BoundRegionalLoraModuleClass.UNSUPPORTED:
            return TargetShapeAdmission(
                module_class,
                None,
                f"Target module type {type(module).__name__} is unsupported.",
            )
        if not isinstance(parameter, (torch.Tensor, nn.Parameter)):
            return TargetShapeAdmission(
                module_class,
                parameter,
                "Target parameter must be a Torch tensor or Parameter.",
            )
        target_shape, offset_issue = _target_shape(parameter, descriptor.target.offset)
        if offset_issue is not None:
            return TargetShapeAdmission(module_class, parameter, offset_issue)
        issue = _shape_issue(descriptor, module, target_shape)
        return TargetShapeAdmission(module_class, parameter, issue)


def _module_class(module: object) -> BoundRegionalLoraModuleClass:
    """Classify exact supported Torch operation families."""

    if isinstance(module, nn.Linear):
        return BoundRegionalLoraModuleClass.LINEAR
    if isinstance(module, nn.Conv1d):
        return BoundRegionalLoraModuleClass.CONVOLUTION_1D
    if isinstance(module, nn.Conv2d):
        return BoundRegionalLoraModuleClass.CONVOLUTION_2D
    if isinstance(module, nn.Conv3d):
        return BoundRegionalLoraModuleClass.CONVOLUTION_3D
    return BoundRegionalLoraModuleClass.UNSUPPORTED


def _target_shape(
    parameter: torch.Tensor,
    offset: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], str | None]:
    """Return the exact sliced parameter shape described by Comfy's target path."""

    shape = tuple(int(dimension) for dimension in parameter.shape)
    if offset is None:
        return shape, None
    if len(offset) != 3:
        return shape, "Comfy target offset must contain dimension, start, and length."
    dimension, start, length = offset
    if dimension >= len(shape):
        return shape, "Comfy target offset dimension exceeds parameter rank."
    if length <= 0 or start + length > shape[dimension]:
        return shape, "Comfy target offset exceeds the parameter dimension."
    sliced = list(shape)
    sliced[dimension] = length
    return tuple(sliced), None


def _shape_issue(
    descriptor: ResolvedRegionalLoraOperation,
    module: object,
    target_shape: tuple[int, ...],
) -> str | None:
    """Validate low-rank composition against the observed operation shape."""

    if descriptor.down_shape is None or descriptor.up_shape is None:
        return "Supported descriptor is missing down/up shapes."
    expected_target_shape = (
        descriptor.reshape_shape.dimensions
        if descriptor.reshape_shape is not None
        else target_shape
    )
    if descriptor.reshape_shape is not None:
        if len(expected_target_shape) != len(target_shape):
            return "Authored reshape rank must match the target parameter rank."
        if any(
            authored < observed
            for authored, observed in zip(
                expected_target_shape, target_shape, strict=True
            )
        ):
            return "Authored reshape cannot shrink an observed target dimension."
    down = descriptor.down_shape.dimensions
    up = descriptor.up_shape.dimensions
    if up[0] != expected_target_shape[0]:
        return "LoRA up output channels do not match the target parameter."
    if isinstance(module, nn.Linear):
        if descriptor.operation_class is not ResolvedLoraOperationClass.MATRIX_PAIR:
            return "Linear target requires a matrix-pair LoRA descriptor."
        if len(expected_target_shape) != 2 or down[1] != expected_target_shape[1]:
            return "LoRA down input features do not match the linear parameter."
        return None
    if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return "Target module type is unsupported."
    convolution_dimension = module.weight.ndim - 2
    expected_class = {
        1: ResolvedLoraOperationClass.CONVOLUTION_1D,
        2: ResolvedLoraOperationClass.CONVOLUTION_2D,
        3: ResolvedLoraOperationClass.CONVOLUTION_3D,
    }[convolution_dimension]
    if descriptor.operation_class not in (
        ResolvedLoraOperationClass.MATRIX_PAIR,
        expected_class,
    ):
        return "LoRA convolution dimensionality does not match the target module."
    flattened_input = math.prod(expected_target_shape[1:])
    if len(down) == 2:
        if down[1] != flattened_input:
            return "Matrix LoRA down input size does not match the convolution weight."
    elif descriptor.middle_shape is None:
        if tuple(down[1:]) != tuple(expected_target_shape[1:]):
            return "Convolutional LoRA down shape does not match the target weight."
    elif (
        down[1] != expected_target_shape[1]
        or any(kernel != 1 for kernel in down[2:])
        or tuple(descriptor.middle_shape.dimensions[2:])
        != tuple(expected_target_shape[2:])
    ):
        return "LoCon down/middle shapes do not compose to the target weight."
    spatial_rank = convolution_dimension + 2
    if len(up) not in (2, spatial_rank):
        return "Convolutional LoRA up tensor dimensionality is incompatible."
    if len(up) == spatial_rank and any(kernel != 1 for kernel in up[2:]):
        return "Convolutional LoRA up projection must use unit spatial kernels."
    middle = descriptor.middle_shape
    if middle is not None and middle.rank not in (2, spatial_rank):
        return "Convolutional LoRA middle dimensionality is incompatible."
    return None


REGIONAL_LORA_TARGET_SHAPE_VALIDATOR = RegionalLoraTargetShapeValidator()
