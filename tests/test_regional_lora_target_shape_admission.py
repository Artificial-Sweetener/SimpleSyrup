# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify regional LoRA descriptor compatibility with real Torch operations."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.domain.resolved_regional_lora import (
    RegionalLoraExecutionContract,
    RegionalLoraGeometryClass,
    RegionalLoraTensorShape,
    ResolvedLoraOperationClass,
    ResolvedRegionalLoraOperation,
    ResolvedRegionalLoraTarget,
)
from simple_syrup.runtime.regional_lora.target_binding import (
    BoundRegionalLoraModuleClass,
)
from simple_syrup.runtime.regional_lora.target_shape_admission import (
    RegionalLoraTargetShapeValidator,
)


def test_validator_admits_linear_matrix_pair() -> None:
    """Match matrix input/output features and exact parameter identity."""

    module = nn.Linear(4, 6, bias=False)

    result = RegionalLoraTargetShapeValidator().validate(
        _matrix_descriptor(input_size=4, output_size=6),
        module,
        module.weight,
    )

    assert result.module_class is BoundRegionalLoraModuleClass.LINEAR
    assert result.parameter is module.weight
    assert result.issue is None


@pytest.mark.parametrize(
    ("module", "operation_class"),
    [
        (nn.Conv1d(4, 6, 3, bias=False), ResolvedLoraOperationClass.CONVOLUTION_1D),
        (nn.Conv2d(4, 6, 3, bias=False), ResolvedLoraOperationClass.CONVOLUTION_2D),
        (nn.Conv3d(4, 6, 3, bias=False), ResolvedLoraOperationClass.CONVOLUTION_3D),
    ],
)
def test_validator_admits_explicit_convolution_dimensions(
    module: nn.Conv1d | nn.Conv2d | nn.Conv3d,
    operation_class: ResolvedLoraOperationClass,
) -> None:
    """Match direct convolution channels, kernels, rank, and module dimension."""

    weight = module.weight
    descriptor = _convolution_descriptor(
        operation_class=operation_class,
        target_shape=tuple(weight.shape),
    )

    result = RegionalLoraTargetShapeValidator().validate(
        descriptor,
        module,
        weight,
    )

    assert result.issue is None


def test_validator_admits_matrix_pair_for_grouped_convolution() -> None:
    """Use the observed grouped weight shape for Comfy's matrix conv form."""

    module = nn.Conv2d(4, 6, 3, groups=2, bias=False)
    descriptor = _matrix_descriptor(
        input_size=2 * 3 * 3,
        output_size=6,
    )

    result = RegionalLoraTargetShapeValidator().validate(
        descriptor,
        module,
        module.weight,
    )

    assert result.issue is None


def test_validator_admits_locon_down_middle_composition() -> None:
    """Match 1x1 down plus spatial middle against the target kernel."""

    module = nn.Conv2d(4, 6, 3, bias=False)
    descriptor = _convolution_descriptor(
        operation_class=ResolvedLoraOperationClass.CONVOLUTION_2D,
        target_shape=tuple(module.weight.shape),
        use_middle=True,
    )

    result = RegionalLoraTargetShapeValidator().validate(
        descriptor,
        module,
        module.weight,
    )

    assert result.issue is None


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("linear-input", "input features"),
        ("linear-output", "output channels"),
        ("convolution-dimension", "dimensionality"),
        ("module", "module type"),
        ("parameter", "Torch tensor"),
    ],
)
def test_validator_rejects_module_parameter_and_shape_mismatch(
    case: str,
    message: str,
) -> None:
    """Return explicit incompatibility without changing the observed module."""

    descriptor, module, parameter = _invalid_case(case)
    result = RegionalLoraTargetShapeValidator().validate(
        descriptor,
        module,
        parameter,
    )

    assert result.issue is not None
    assert message in result.issue


def _invalid_case(
    case: str,
) -> tuple[ResolvedRegionalLoraOperation, nn.Module, object]:
    """Construct one focused invalid admission case after module import."""

    if case == "linear-input":
        return (
            _matrix_descriptor(input_size=5, output_size=6),
            nn.Linear(4, 6, bias=False),
            torch.ones((6, 4)),
        )
    if case == "linear-output":
        return (
            _matrix_descriptor(input_size=4, output_size=7),
            nn.Linear(4, 6, bias=False),
            torch.ones((6, 4)),
        )
    if case == "convolution-dimension":
        return (
            _convolution_descriptor(
                operation_class=ResolvedLoraOperationClass.CONVOLUTION_1D,
                target_shape=(6, 4, 3),
            ),
            nn.Conv2d(4, 6, 3, bias=False),
            torch.ones((6, 4, 3, 3)),
        )
    if case == "module":
        return (
            _matrix_descriptor(input_size=4, output_size=6),
            nn.ReLU(),
            torch.ones((6, 4)),
        )
    if case == "parameter":
        return (
            _matrix_descriptor(input_size=4, output_size=6),
            nn.Linear(4, 6, bias=False),
            object(),
        )
    raise AssertionError(f"Unknown invalid admission case: {case}")


def test_validator_applies_comfy_offset_to_target_shape() -> None:
    """Validate one exact sliced target shape without narrowing the parameter."""

    module = nn.Linear(4, 12, bias=False)
    descriptor = replace(
        _matrix_descriptor(input_size=4, output_size=6),
        target=ResolvedRegionalLoraTarget("diffusion_model.layer", "weight", (0, 3, 6)),
    )
    before = (
        module.weight.data_ptr(),
        module.weight._version,
        tuple(module.weight.shape),
    )

    result = RegionalLoraTargetShapeValidator().validate(
        descriptor,
        module,
        module.weight,
    )

    assert result.issue is None
    assert before == (
        module.weight.data_ptr(),
        module.weight._version,
        tuple(module.weight.shape),
    )


def _matrix_descriptor(
    *,
    input_size: int,
    output_size: int,
) -> ResolvedRegionalLoraOperation:
    """Return one valid matrix-pair descriptor."""

    return _descriptor(
        operation_class=ResolvedLoraOperationClass.MATRIX_PAIR,
        down_shape=RegionalLoraTensorShape((2, input_size)),
        up_shape=RegionalLoraTensorShape((output_size, 2)),
        middle_shape=None,
        geometry=RegionalLoraGeometryClass.TARGET_OPERATION,
        execution=RegionalLoraExecutionContract.MATRIX_OR_RESHAPED_CONVOLUTION,
    )


def _convolution_descriptor(
    *,
    operation_class: ResolvedLoraOperationClass,
    target_shape: tuple[int, ...],
    use_middle: bool = False,
) -> ResolvedRegionalLoraOperation:
    """Return one valid direct convolution descriptor."""

    spatial = target_shape[2:]
    down_spatial = (1,) * len(spatial) if use_middle else spatial
    middle = RegionalLoraTensorShape((2, 2, *spatial)) if use_middle else None
    geometry = {
        1: RegionalLoraGeometryClass.DIRECT_CONVOLUTION_1D,
        2: RegionalLoraGeometryClass.DIRECT_CONVOLUTION_2D,
        3: RegionalLoraGeometryClass.DIRECT_CONVOLUTION_3D,
    }[len(spatial)]
    return _descriptor(
        operation_class=operation_class,
        down_shape=RegionalLoraTensorShape((2, target_shape[1], *down_spatial)),
        up_shape=RegionalLoraTensorShape((target_shape[0], 2)),
        middle_shape=middle,
        geometry=geometry,
        execution=RegionalLoraExecutionContract.DIRECT_CONVOLUTION,
    )


def _descriptor(
    *,
    operation_class: ResolvedLoraOperationClass,
    down_shape: RegionalLoraTensorShape,
    up_shape: RegionalLoraTensorShape,
    middle_shape: RegionalLoraTensorShape | None,
    geometry: RegionalLoraGeometryClass,
    execution: RegionalLoraExecutionContract,
) -> ResolvedRegionalLoraOperation:
    """Return one shared valid descriptor shell."""

    return ResolvedRegionalLoraOperation(
        adapter=RegionalLoraAdapterPlan(
            RegionalLoraAdapterIdentity("adapter.safetensors"),
            0,
            0,
            RegionalLoraBranch.POSITIVE,
            1.0,
            (RegionalLoraScheduleBoundary(0.0, 1.0, 1.0, 0),),
        ),
        target_index=0,
        target=ResolvedRegionalLoraTarget("diffusion_model.layer", "weight", None),
        normalized_operation_type="LoRAAdapter",
        operation_class=operation_class,
        down_shape=down_shape,
        up_shape=up_shape,
        middle_shape=middle_shape,
        reshape_shape=None,
        rank=2,
        intrinsic_scale=1.0,
        required_geometry=geometry,
        execution_contract=execution,
        rejection_reason=None,
    )
