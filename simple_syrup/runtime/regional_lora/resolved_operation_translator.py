# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Translate normalized Comfy LoRA targets into model-neutral descriptors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from comfy.weight_adapter.lora import LoRAAdapter

from ...domain.regional_lora_plan import RegionalLoraAdapterPlan
from ...domain.resolved_regional_lora import (
    RegionalLoraExecutionContract,
    RegionalLoraGeometryClass,
    RegionalLoraTensorShape,
    ResolvedLoraOperationClass,
    ResolvedRegionalLoraOperation,
    ResolvedRegionalLoraOperationSet,
    ResolvedRegionalLoraTarget,
)
from .comfy_adapter_resolution import (
    ComfyNormalizedAdapterTarget,
    ComfyRegionalLoraResolution,
)


@dataclass(frozen=True, slots=True)
class _OrdinaryLoraMetadata:
    """Retain validated descriptor metadata during one translation."""

    operation_class: ResolvedLoraOperationClass
    down_shape: RegionalLoraTensorShape
    up_shape: RegionalLoraTensorShape
    middle_shape: RegionalLoraTensorShape | None
    reshape_shape: RegionalLoraTensorShape | None
    rank: int
    intrinsic_scale: float
    required_geometry: RegionalLoraGeometryClass
    execution_contract: RegionalLoraExecutionContract


class ComfyResolvedOperationTranslator:
    """Translate U2 results without model binding, tensor movement, or mutation."""

    def translate(
        self,
        resolution: ComfyRegionalLoraResolution,
    ) -> ResolvedRegionalLoraOperationSet:
        """Return one descriptor for every normalized target in declared order."""

        return ResolvedRegionalLoraOperationSet(
            tuple(
                self._translate_target(adapter_result.adapter, target_index, target)
                for adapter_result in resolution.adapters
                for target_index, target in enumerate(adapter_result.model_targets)
            )
        )

    def _translate_target(
        self,
        adapter: RegionalLoraAdapterPlan,
        target_index: int,
        target: ComfyNormalizedAdapterTarget,
    ) -> ResolvedRegionalLoraOperation:
        """Translate one target or retain its exact rejection reason."""

        try:
            resolved_target = _resolved_target(target)
            metadata = _ordinary_lora_metadata(target)
            return ResolvedRegionalLoraOperation(
                adapter=adapter,
                target_index=target_index,
                target=resolved_target,
                normalized_operation_type=target.operation_type,
                operation_class=metadata.operation_class,
                down_shape=metadata.down_shape,
                up_shape=metadata.up_shape,
                middle_shape=metadata.middle_shape,
                reshape_shape=metadata.reshape_shape,
                rank=metadata.rank,
                intrinsic_scale=metadata.intrinsic_scale,
                required_geometry=metadata.required_geometry,
                execution_contract=metadata.execution_contract,
                rejection_reason=None,
            )
        except (IndexError, TypeError, ValueError) as error:
            return _rejection(
                adapter=adapter,
                target_index=target_index,
                target=_fallback_target(target),
                normalized_operation_type=target.operation_type,
                reason=str(error),
            )


def _ordinary_lora_metadata(
    target: ComfyNormalizedAdapterTarget,
) -> _OrdinaryLoraMetadata:
    """Validate and classify one exact installed Comfy normalized operation."""

    if not isinstance(target.operation, LoRAAdapter):
        raise ValueError(
            f"Normalized operation {target.operation_type} is not an ordinary LoRA."
        )
    weights = target.operation.weights
    if not isinstance(weights, (tuple, list)) or len(weights) != 6:
        raise ValueError("Normalized LoRA operation must contain six weight fields.")
    up, down, alpha, middle, dora_scale, reshape = weights
    if dora_scale is not None:
        raise ValueError("DoRA scaling is unsupported.")
    if not target.ordinary_additive_lora:
        raise ValueError("Normalized LoRA target is not ordinary additive LoRA.")
    up_tensor = _require_tensor(up, name="up")
    down_tensor = _require_tensor(down, name="down")
    middle_tensor = (
        _require_tensor(middle, name="middle") if middle is not None else None
    )
    up_shape = _shape(up_tensor)
    down_shape = _shape(down_tensor)
    middle_shape = _shape(middle_tensor) if middle_tensor is not None else None
    rank = down_shape.dimensions[0]
    intrinsic_scale = _intrinsic_scale(alpha, rank=rank)
    reshape_shape = _reshape_shape(reshape)
    operation_class, geometry, execution = _operation_contract(
        down_shape,
        up_shape,
        middle_shape,
    )
    return _OrdinaryLoraMetadata(
        operation_class=operation_class,
        down_shape=down_shape,
        up_shape=up_shape,
        middle_shape=middle_shape,
        reshape_shape=reshape_shape,
        rank=rank,
        intrinsic_scale=intrinsic_scale,
        required_geometry=geometry,
        execution_contract=execution,
    )


def _operation_contract(
    down_shape: RegionalLoraTensorShape,
    up_shape: RegionalLoraTensorShape,
    middle_shape: RegionalLoraTensorShape | None,
) -> tuple[
    ResolvedLoraOperationClass,
    RegionalLoraGeometryClass,
    RegionalLoraExecutionContract,
]:
    """Classify tensor organization without claiming a target module class."""

    shapes = tuple(
        shape for shape in (down_shape, up_shape, middle_shape) if shape is not None
    )
    maximum_rank = max(shape.rank for shape in shapes)
    if maximum_rank == 2 and middle_shape is None:
        return (
            ResolvedLoraOperationClass.MATRIX_PAIR,
            RegionalLoraGeometryClass.TARGET_OPERATION,
            RegionalLoraExecutionContract.MATRIX_OR_RESHAPED_CONVOLUTION,
        )
    contracts = {
        3: (
            ResolvedLoraOperationClass.CONVOLUTION_1D,
            RegionalLoraGeometryClass.DIRECT_CONVOLUTION_1D,
        ),
        4: (
            ResolvedLoraOperationClass.CONVOLUTION_2D,
            RegionalLoraGeometryClass.DIRECT_CONVOLUTION_2D,
        ),
        5: (
            ResolvedLoraOperationClass.CONVOLUTION_3D,
            RegionalLoraGeometryClass.DIRECT_CONVOLUTION_3D,
        ),
    }
    if maximum_rank not in contracts:
        raise ValueError(
            "Normalized LoRA tensor dimensionality must describe a matrix pair "
            "or Conv1d/2d/3d operation."
        )
    if any(shape.rank not in (2, maximum_rank) for shape in shapes):
        raise ValueError("Normalized convolutional LoRA tensor ranks are inconsistent.")
    operation_class, geometry = contracts[maximum_rank]
    return (
        operation_class,
        geometry,
        RegionalLoraExecutionContract.DIRECT_CONVOLUTION,
    )


def _resolved_target(
    target: ComfyNormalizedAdapterTarget,
) -> ResolvedRegionalLoraTarget:
    """Split Comfy's parameter path without resolving it against a model."""

    model_target, separator, parameter_name = target.path.parameter_key.rpartition(".")
    if not separator or not model_target or not parameter_name:
        raise ValueError(
            "Normalized regional LoRA path must identify a model target and parameter."
        )
    return ResolvedRegionalLoraTarget(
        model_target=model_target,
        parameter_name=parameter_name,
        offset=target.path.offset,
    )


def _fallback_target(
    target: ComfyNormalizedAdapterTarget,
) -> ResolvedRegionalLoraTarget:
    """Retain a malformed path diagnostically inside a rejected descriptor."""

    return ResolvedRegionalLoraTarget(
        model_target=target.path.parameter_key or "<empty-target>",
        parameter_name="<invalid-parameter>",
        offset=target.path.offset,
    )


def _require_tensor(value: object, *, name: str) -> torch.Tensor:
    """Require tensor metadata without moving, cloning, or reading its values."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Normalized LoRA {name} weight must be a tensor.")
    return value


def _shape(tensor: torch.Tensor) -> RegionalLoraTensorShape:
    """Copy immutable integer shape metadata from one tensor."""

    return RegionalLoraTensorShape(tuple(int(dimension) for dimension in tensor.shape))


def _intrinsic_scale(alpha: object, *, rank: int) -> float:
    """Apply Comfy's exact optional alpha-over-rank ordinary-LoRA scaling."""

    if alpha is None:
        return 1.0
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("Normalized LoRA alpha must be a finite scalar or None.")
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value):
        raise ValueError("Normalized LoRA alpha must be finite.")
    return alpha_value / rank


def _reshape_shape(value: object) -> RegionalLoraTensorShape | None:
    """Copy Comfy's optional authored target reshape metadata immutably."""

    if value is None:
        return None
    if not isinstance(value, (tuple, list)):
        raise TypeError("Normalized LoRA reshape metadata must be a sequence.")
    dimensions: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError(
                "Normalized LoRA reshape dimensions must be positive integers."
            )
        dimensions.append(dimension)
    return RegionalLoraTensorShape(tuple(dimensions))


def _rejection(
    *,
    adapter: RegionalLoraAdapterPlan,
    target_index: int,
    target: ResolvedRegionalLoraTarget,
    normalized_operation_type: str,
    reason: str,
) -> ResolvedRegionalLoraOperation:
    """Return one complete rejection without partial executable metadata."""

    return ResolvedRegionalLoraOperation(
        adapter=adapter,
        target_index=target_index,
        target=target,
        normalized_operation_type=normalized_operation_type,
        operation_class=ResolvedLoraOperationClass.UNSUPPORTED,
        down_shape=None,
        up_shape=None,
        middle_shape=None,
        reshape_shape=None,
        rank=None,
        intrinsic_scale=None,
        required_geometry=RegionalLoraGeometryClass.UNSUPPORTED,
        execution_contract=RegionalLoraExecutionContract.UNSUPPORTED,
        rejection_reason=reason,
    )


COMFY_RESOLVED_OPERATION_TRANSLATOR = ComfyResolvedOperationTranslator()
