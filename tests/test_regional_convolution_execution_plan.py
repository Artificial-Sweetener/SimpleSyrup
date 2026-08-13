# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify convolution plans adapt exact bound Comfy operation evidence."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import comfy.model_patcher
import pytest
import torch
from comfy.weight_adapter.lora import LoRAAdapter
from torch import nn

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.runtime.regional_lora.comfy_adapter_resolution import (
    ComfyAdapterTargetPath,
    ComfyNormalizedAdapterTarget,
    ComfyRegionalAdapterResolution,
    ComfyRegionalLoraResolution,
)
from simple_syrup.runtime.regional_lora.convolution_execution_plan import (
    RegionalConvolutionExecutionPlanFactory,
)
from simple_syrup.runtime.regional_lora.resolved_operation_translator import (
    ComfyResolvedOperationTranslator,
)
from simple_syrup.runtime.regional_lora.target_binder import RegionalLoraTargetBinder
from simple_syrup.runtime.regional_lora.target_binding import (
    RegionalLoraBindingResult,
)
from simple_syrup.runtime.regional_lora_host_payload import RegionalLoraHostPayload


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_factory_preserves_bound_dimension_parameters_and_tensor_identity(
    dimension: int,
) -> None:
    """Adapt real bound Conv1d/2d/3d evidence without copying normalized tensors."""

    patcher, module = _patcher(dimension)
    target = _target(dimension)
    binding = _binding(patcher, target)

    plan = RegionalConvolutionExecutionPlanFactory().build(binding.entries)

    operation = target.operation
    assert isinstance(operation, LoRAAdapter)
    up, down, *_rest = operation.weights
    assert plan.uses[0].preparation.down is down
    assert plan.uses[0].preparation.up is up
    assert plan.uses[0].parameters.dimension == dimension
    assert plan.uses[0].parameters.stride == tuple(module.stride)
    assert plan.uses[0].parameters.padding == tuple(module.padding)
    assert plan.uses[0].parameters.dilation == tuple(module.dilation)
    assert plan.uses[0].parameters.kernel_size == tuple(module.kernel_size)


def test_factory_rejects_nonzero_padding_mode_before_execution() -> None:
    """Fail before sampling when functional down cannot match the original boundary."""

    patcher, _module = _patcher(2, padding_mode="reflect")
    binding = _binding(patcher, _target(2))

    with pytest.raises(ValueError, match="zero padding mode"):
        RegionalConvolutionExecutionPlanFactory().build(binding.entries)


def test_factory_rejects_malformed_pointwise_up_before_execution() -> None:
    """Reject a bound-evidence mutation that cannot be an exact pointwise up path."""

    patcher, _module = _patcher(2)
    target = _target(2)
    binding = _binding(patcher, target)
    entry = binding.entries[0]
    operation = target.operation
    assert isinstance(operation, LoRAAdapter)
    _up, down, alpha, middle, dora, reshape_shape = operation.weights
    malformed = LoRAAdapter(
        {"up", "down"},
        (torch.ones((3, 2, 2, 2)), down, alpha, middle, dora, reshape_shape),
    )
    normalized = replace(target, operation=malformed)
    altered = replace(entry, normalized_target=normalized)

    with pytest.raises(ValueError, match="pointwise"):
        RegionalConvolutionExecutionPlanFactory().build((altered,))


def _patcher(
    dimension: int,
    *,
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
) -> tuple[comfy.model_patcher.ModelPatcher, nn.Conv1d | nn.Conv2d | nn.Conv3d]:
    """Return one real patcher around an observed dimensional convolution."""

    convolution = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[dimension - 1]
    module = convolution(
        2,
        3,
        3,
        stride=2,
        padding=1,
        dilation=1,
        padding_mode=padding_mode,
        bias=False,
    )
    root = nn.Module()
    root.diffusion_model = nn.Module()
    root.diffusion_model.layer = module
    return (
        comfy.model_patcher.ModelPatcher(
            root,
            torch.device("cpu"),
            torch.device("cpu"),
        ),
        module,
    )


def _target(dimension: int) -> ComfyNormalizedAdapterTarget:
    """Return one exact normalized direct dimensional LoRA target."""

    spatial = (3,) * dimension
    return ComfyNormalizedAdapterTarget(
        ComfyAdapterTargetPath("diffusion_model.layer.weight", None),
        LoRAAdapter(
            {"up", "down"},
            (
                torch.ones((3, 2, *((1,) * dimension))),
                torch.ones((2, 2, *spatial)),
                None,
                None,
                None,
                None,
            ),
        ),
        "LoRAAdapter",
        ("down", "up"),
        True,
    )


def _binding(
    patcher: comfy.model_patcher.ModelPatcher,
    target: ComfyNormalizedAdapterTarget,
) -> RegionalLoraBindingResult:
    """Resolve, translate, and bind one convolution target through U2-U4 values."""

    adapter = RegionalLoraAdapterPlan(
        RegionalLoraAdapterIdentity("conv.safetensors"),
        0,
        0,
        RegionalLoraBranch.POSITIVE,
        0.8,
        (RegionalLoraScheduleBoundary(0.0, 1.0, 1.0, 0),),
    )
    resolution = ComfyRegionalLoraResolution(
        (
            ComfyRegionalAdapterResolution(
                adapter,
                RegionalLoraHostPayload(False, None, {}, None),
                (target,),
                (),
                (),
            ),
        ),
        (),
    )
    operations = ComfyResolvedOperationTranslator().translate(resolution)
    return RegionalLoraTargetBinder().bind(
        source=patcher,
        candidate=patcher,
        resolution=resolution,
        operations=operations,
    )
