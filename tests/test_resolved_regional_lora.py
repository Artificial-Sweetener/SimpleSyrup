# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify immutable model-neutral resolved regional LoRA contracts."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import pytest

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
    ResolvedRegionalLoraOperationSet,
    ResolvedRegionalLoraTarget,
)


def test_matrix_pair_retains_plan_target_shapes_scale_and_strength() -> None:
    """Represent a matrix pair without guessing its eventual module class."""

    adapter = _adapter(0)
    operation = _matrix_operation(adapter=adapter, target_index=0)

    assert operation.adapter is adapter
    assert operation.target.parameter_path == "diffusion_model.layer.weight"
    assert operation.target.offset is None
    assert operation.down_shape == RegionalLoraTensorShape((4, 16))
    assert operation.up_shape == RegionalLoraTensorShape((32, 4))
    assert operation.rank == 4
    assert operation.intrinsic_scale == 0.5
    assert operation.authored_strength == 0.8
    assert operation.required_geometry is RegionalLoraGeometryClass.TARGET_OPERATION
    assert operation.supported is True


@pytest.mark.parametrize(
    ("operation_class", "tensor_rank", "geometry"),
    [
        (
            ResolvedLoraOperationClass.CONVOLUTION_1D,
            3,
            RegionalLoraGeometryClass.DIRECT_CONVOLUTION_1D,
        ),
        (
            ResolvedLoraOperationClass.CONVOLUTION_2D,
            4,
            RegionalLoraGeometryClass.DIRECT_CONVOLUTION_2D,
        ),
        (
            ResolvedLoraOperationClass.CONVOLUTION_3D,
            5,
            RegionalLoraGeometryClass.DIRECT_CONVOLUTION_3D,
        ),
    ],
)
def test_convolution_contracts_represent_direct_and_mixed_tensor_shapes(
    operation_class: ResolvedLoraOperationClass,
    tensor_rank: int,
    geometry: RegionalLoraGeometryClass,
) -> None:
    """Represent direct convolution while allowing Comfy's 2D up projection."""

    spatial = (3,) * (tensor_rank - 2)
    operation = ResolvedRegionalLoraOperation(
        adapter=_adapter(0),
        target_index=0,
        target=_target(),
        normalized_operation_type="LoRAAdapter",
        operation_class=operation_class,
        down_shape=RegionalLoraTensorShape((4, 16, *spatial)),
        up_shape=RegionalLoraTensorShape((32, 4)),
        middle_shape=RegionalLoraTensorShape((4, 4, *spatial)),
        reshape_shape=None,
        rank=4,
        intrinsic_scale=1.0,
        required_geometry=geometry,
        execution_contract=RegionalLoraExecutionContract.DIRECT_CONVOLUTION,
        rejection_reason=None,
    )

    assert operation.supported is True
    assert operation.middle_shape is not None
    assert operation.middle_shape.rank == tensor_rank


def test_rejection_requires_reason_and_cannot_claim_executable_metadata() -> None:
    """Keep unsupported targets explicit without misleading partial math."""

    rejected = _rejection(adapter=_adapter(0), target_index=0)

    assert rejected.supported is False
    assert rejected.rejection_reason == "DoRA scaling is unsupported."
    assert rejected.down_shape is None

    with pytest.raises(ValueError, match="requires a reason"):
        replace(rejected, rejection_reason="")
    with pytest.raises(ValueError, match="cannot claim executable metadata"):
        replace(rejected, rank=4)


@pytest.mark.parametrize(
    ("dimensions", "message"),
    [
        ((), "nonempty tuple"),
        ((0, 2), "positive integers"),
        ((-1, 2), "positive integers"),
        ((cast(Any, True), 2), "positive integers"),
        ((cast(Any, 1.5), 2), "positive integers"),
    ],
)
def test_tensor_shape_rejects_mutable_or_invalid_dimensions(
    dimensions: tuple[int, ...],
    message: str,
) -> None:
    """Require immutable positive integer shape metadata."""

    with pytest.raises(ValueError, match=message):
        RegionalLoraTensorShape(dimensions)
    with pytest.raises(ValueError, match="nonempty tuple"):
        RegionalLoraTensorShape(cast(Any, [2, 2]))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda operation: replace(operation, rank=3), "down shape must begin"),
        (
            lambda operation: replace(
                operation,
                up_shape=RegionalLoraTensorShape((32, 3)),
            ),
            "rank at index one",
        ),
        (
            lambda operation: replace(operation, intrinsic_scale=float("nan")),
            "intrinsic scale must be finite",
        ),
        (
            lambda operation: replace(
                operation,
                intrinsic_scale=cast(Any, 1),
            ),
            "intrinsic scale must be finite",
        ),
        (
            lambda operation: replace(
                operation,
                middle_shape=RegionalLoraTensorShape((4,)),
            ),
            "at least two",
        ),
        (
            lambda operation: replace(
                operation,
                middle_shape=RegionalLoraTensorShape((4, 4, 1, 1)),
            ),
            "cannot contain middle",
        ),
        (
            lambda operation: replace(
                operation,
                required_geometry=RegionalLoraGeometryClass.DIRECT_CONVOLUTION_2D,
            ),
            "requires target geometry",
        ),
    ],
)
def test_supported_operation_rejects_inconsistent_math(
    mutation: Callable[
        [ResolvedRegionalLoraOperation],
        ResolvedRegionalLoraOperation,
    ],
    message: str,
) -> None:
    """Reject inconsistent rank, scale, middle, and geometry metadata."""

    with pytest.raises(ValueError, match=message):
        mutation(_matrix_operation(adapter=_adapter(0), target_index=0))


def test_operation_set_preserves_canonical_order_and_partitions_results() -> None:
    """Keep adapter composition and per-adapter target indices contiguous."""

    first = _matrix_operation(adapter=_adapter(0), target_index=0)
    second = _rejection(adapter=_adapter(0), target_index=1)
    third = _matrix_operation(adapter=_adapter(1), target_index=0)
    operation_set = ResolvedRegionalLoraOperationSet((first, second, third))

    assert operation_set.supported == (first, third)
    assert operation_set.rejected == (second,)

    with pytest.raises(ValueError, match="retain declared order"):
        ResolvedRegionalLoraOperationSet((third, first))
    with pytest.raises(ValueError, match="contiguous per adapter"):
        ResolvedRegionalLoraOperationSet((replace(first, target_index=1),))
    with pytest.raises(TypeError, match="must be a tuple"):
        ResolvedRegionalLoraOperationSet(cast(Any, [first]))


def test_domain_values_are_frozen_and_import_no_runtime_frameworks() -> None:
    """Keep mutation and Comfy/Torch dependencies outside the domain owner."""

    operation = _matrix_operation(adapter=_adapter(0), target_index=0)
    with pytest.raises(FrozenInstanceError):
        operation.rank = 8  # type: ignore[misc]

    source = (
        Path(__file__).parents[1]
        / "simple_syrup"
        / "domain"
        / "resolved_regional_lora.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").lstrip(".").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imported_roots.isdisjoint({"comfy", "torch"})


def _matrix_operation(
    *,
    adapter: RegionalLoraAdapterPlan,
    target_index: int,
) -> ResolvedRegionalLoraOperation:
    """Return one valid matrix-pair descriptor."""

    return ResolvedRegionalLoraOperation(
        adapter=adapter,
        target_index=target_index,
        target=_target(),
        normalized_operation_type="LoRAAdapter",
        operation_class=ResolvedLoraOperationClass.MATRIX_PAIR,
        down_shape=RegionalLoraTensorShape((4, 16)),
        up_shape=RegionalLoraTensorShape((32, 4)),
        middle_shape=None,
        reshape_shape=None,
        rank=4,
        intrinsic_scale=0.5,
        required_geometry=RegionalLoraGeometryClass.TARGET_OPERATION,
        execution_contract=(
            RegionalLoraExecutionContract.MATRIX_OR_RESHAPED_CONVOLUTION
        ),
        rejection_reason=None,
    )


def _rejection(
    *,
    adapter: RegionalLoraAdapterPlan,
    target_index: int,
) -> ResolvedRegionalLoraOperation:
    """Return one valid explicit rejection descriptor."""

    return ResolvedRegionalLoraOperation(
        adapter=adapter,
        target_index=target_index,
        target=_target(),
        normalized_operation_type="LoRAAdapter",
        operation_class=ResolvedLoraOperationClass.UNSUPPORTED,
        down_shape=None,
        up_shape=None,
        middle_shape=None,
        reshape_shape=None,
        rank=None,
        intrinsic_scale=None,
        required_geometry=RegionalLoraGeometryClass.UNSUPPORTED,
        execution_contract=RegionalLoraExecutionContract.UNSUPPORTED,
        rejection_reason="DoRA scaling is unsupported.",
    )


def _target() -> ResolvedRegionalLoraTarget:
    """Return one valid model-relative target."""

    return ResolvedRegionalLoraTarget(
        model_target="diffusion_model.layer",
        parameter_name="weight",
        offset=None,
    )


def _adapter(composition_index: int) -> RegionalLoraAdapterPlan:
    """Return one valid authored adapter plan."""

    return RegionalLoraAdapterPlan(
        adapter_identity=RegionalLoraAdapterIdentity(
            f"adapter-{composition_index}.safetensors"
        ),
        composition_index=composition_index,
        region_index=composition_index,
        branch=RegionalLoraBranch.POSITIVE,
        model_strength=0.8,
        schedule=(RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),),
    )
