# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify complete translation from Comfy targets to domain descriptors."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from comfy.weight_adapter.lora import LoRAAdapter

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.domain.resolved_regional_lora import (
    RegionalLoraExecutionContract,
    RegionalLoraGeometryClass,
    ResolvedLoraOperationClass,
)
from simple_syrup.runtime.regional_lora.comfy_adapter_resolution import (
    ComfyAdapterTargetPath,
    ComfyNormalizedAdapterTarget,
    ComfyRegionalAdapterResolution,
    ComfyRegionalLoraResolution,
)
from simple_syrup.runtime.regional_lora.resolved_operation_translator import (
    ComfyResolvedOperationTranslator,
)
from simple_syrup.runtime.regional_lora_host_payload import RegionalLoraHostPayload


def test_translator_retains_matrix_metadata_order_and_authored_plan() -> None:
    """Translate every target without copying or reordering authored plan state."""

    first_adapter = _adapter(0)
    second_adapter = _adapter(1)
    first = _target(
        "diffusion_model.first.weight",
        up=torch.ones((32, 4)),
        down=torch.ones((4, 16)),
        alpha=2.0,
        reshape=[32, 16],
    )
    second = _target(
        "diffusion_model.second.weight",
        up=torch.ones((24, 8)),
        down=torch.ones((8, 12)),
        alpha=None,
        offset=(0, 4, 8),
    )

    translated = ComfyResolvedOperationTranslator().translate(
        _resolution(((first_adapter, (first,)), (second_adapter, (second,))))
    )

    assert [entry.adapter for entry in translated.entries] == [
        first_adapter,
        second_adapter,
    ]
    assert [entry.target_index for entry in translated.entries] == [0, 0]
    assert [entry.target.parameter_path for entry in translated.entries] == [
        "diffusion_model.first.weight",
        "diffusion_model.second.weight",
    ]
    assert translated.entries[0].rank == 4
    assert translated.entries[0].intrinsic_scale == 0.5
    assert translated.entries[0].reshape_shape is not None
    assert translated.entries[0].reshape_shape.dimensions == (32, 16)
    assert translated.entries[1].intrinsic_scale == 1.0
    assert translated.entries[1].target.offset == (0, 4, 8)
    assert all(
        entry.operation_class is ResolvedLoraOperationClass.MATRIX_PAIR
        for entry in translated.entries
    )


@pytest.mark.parametrize(
    ("tensor_rank", "operation_class", "geometry"),
    [
        (
            3,
            ResolvedLoraOperationClass.CONVOLUTION_1D,
            RegionalLoraGeometryClass.DIRECT_CONVOLUTION_1D,
        ),
        (
            4,
            ResolvedLoraOperationClass.CONVOLUTION_2D,
            RegionalLoraGeometryClass.DIRECT_CONVOLUTION_2D,
        ),
        (
            5,
            ResolvedLoraOperationClass.CONVOLUTION_3D,
            RegionalLoraGeometryClass.DIRECT_CONVOLUTION_3D,
        ),
    ],
)
def test_translator_classifies_explicit_convolution_and_middle_weights(
    tensor_rank: int,
    operation_class: ResolvedLoraOperationClass,
    geometry: RegionalLoraGeometryClass,
) -> None:
    """Classify direct convolution solely from normalized tensor organization."""

    spatial = (3,) * (tensor_rank - 2)
    target = _target(
        "diffusion_model.conv.weight",
        up=torch.ones((32, 4)),
        down=torch.ones((4, 16, *spatial)),
        middle=torch.ones((4, 4, *spatial)),
        alpha=4.0,
    )

    entry = (
        ComfyResolvedOperationTranslator()
        .translate(_resolution(((_adapter(0), (target,)),)))
        .entries[0]
    )

    assert entry.operation_class is operation_class
    assert entry.required_geometry is geometry
    assert entry.execution_contract is RegionalLoraExecutionContract.DIRECT_CONVOLUTION
    assert entry.intrinsic_scale == 1.0


def test_translator_rejects_every_unsupported_target_without_omission() -> None:
    """Retain DoRA, non-LoRA, malformed-shape, and malformed-path rejections."""

    ordinary = _target(
        "diffusion_model.valid.weight",
        up=torch.ones((8, 2)),
        down=torch.ones((2, 8)),
    )
    dora = _target(
        "diffusion_model.dora.weight",
        up=torch.ones((8, 2)),
        down=torch.ones((2, 8)),
        dora_scale=torch.ones(8),
    )
    non_lora = ComfyNormalizedAdapterTarget(
        path=ComfyAdapterTargetPath("diffusion_model.diff.weight", None),
        operation=("diff", (torch.ones(1),)),
        operation_type="tuple",
        source_keys=(),
        ordinary_additive_lora=False,
    )
    bad_rank = _target(
        "diffusion_model.bad.weight",
        up=torch.ones((8, 3)),
        down=torch.ones((2, 8)),
    )
    malformed_path = _target(
        "weight",
        up=torch.ones((8, 2)),
        down=torch.ones((2, 8)),
    )

    result = ComfyResolvedOperationTranslator().translate(
        _resolution(
            ((_adapter(0), (ordinary, dora, non_lora, bad_rank, malformed_path)),)
        )
    )

    assert len(result.entries) == 5
    assert result.supported == (result.entries[0],)
    assert result.rejected == result.entries[1:]
    assert [entry.target_index for entry in result.entries] == [0, 1, 2, 3, 4]
    assert [entry.rejection_reason for entry in result.rejected] == [
        "DoRA scaling is unsupported.",
        "Normalized operation tuple is not an ordinary LoRA.",
        "Regional LoRA up shape must contain its rank at index one.",
        ("Normalized regional LoRA path must identify a model target and parameter."),
    ]


@pytest.mark.parametrize(
    ("alpha", "reshape", "message"),
    [
        (float("inf"), None, "alpha must be finite"),
        (cast(float, True), None, "finite scalar or None"),
        (1.0, [8, 0], "positive integers"),
        (1.0, [8, cast(int, True)], "positive integers"),
    ],
)
def test_translator_rejects_invalid_alpha_and_reshape_metadata(
    alpha: float,
    reshape: list[int] | None,
    message: str,
) -> None:
    """Convert malformed normalized metadata into explicit target rejections."""

    target = _target(
        "diffusion_model.layer.weight",
        up=torch.ones((8, 2)),
        down=torch.ones((2, 8)),
        alpha=alpha,
        reshape=reshape,
    )

    entry = (
        ComfyResolvedOperationTranslator()
        .translate(_resolution(((_adapter(0), (target,)),)))
        .entries[0]
    )

    assert entry.supported is False
    assert entry.rejection_reason is not None
    assert message in entry.rejection_reason


def test_translator_reads_metadata_without_moving_or_mutating_tensors() -> None:
    """Preserve tensor identity and state during translation."""

    up = torch.ones((8, 2), dtype=torch.float16)
    down = torch.ones((2, 8), dtype=torch.float16)
    before = tuple(
        (
            id(tensor),
            tensor.data_ptr(),
            tensor._version,
            tensor.device,
            tensor.dtype,
            tensor.shape,
        )
        for tensor in (up, down)
    )
    target = _target("diffusion_model.layer.weight", up=up, down=down)

    result = ComfyResolvedOperationTranslator().translate(
        _resolution(((_adapter(0), (target,)),))
    )

    assert result.entries[0].supported is True
    assert before == tuple(
        (
            id(tensor),
            tensor.data_ptr(),
            tensor._version,
            tensor.device,
            tensor.dtype,
            tensor.shape,
        )
        for tensor in (up, down)
    )


def _resolution(
    adapters: tuple[
        tuple[RegionalLoraAdapterPlan, tuple[ComfyNormalizedAdapterTarget, ...]], ...
    ],
) -> ComfyRegionalLoraResolution:
    """Return one ordered U2 resolution for translator characterization."""

    results = tuple(
        ComfyRegionalAdapterResolution(
            adapter=adapter,
            payload=RegionalLoraHostPayload(False, None, {}, None),
            model_targets=targets,
            source_entries=(),
            issues=(),
        )
        for adapter, targets in adapters
    )
    return ComfyRegionalLoraResolution(adapters=results, issues=())


def _target(
    parameter_path: str,
    *,
    up: torch.Tensor,
    down: torch.Tensor,
    alpha: float | None = None,
    middle: torch.Tensor | None = None,
    dora_scale: torch.Tensor | None = None,
    reshape: list[int] | None = None,
    offset: tuple[int, ...] | None = None,
) -> ComfyNormalizedAdapterTarget:
    """Return one installed normalized LoRA target with exact tensor references."""

    operation = LoRAAdapter(
        {"up", "down"},
        (up, down, alpha, middle, dora_scale, reshape),
    )
    return ComfyNormalizedAdapterTarget(
        path=ComfyAdapterTargetPath(parameter_path, offset),
        operation=operation,
        operation_type="LoRAAdapter",
        source_keys=("down", "up"),
        ordinary_additive_lora=dora_scale is None,
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
