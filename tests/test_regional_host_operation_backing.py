# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove lightweight regional wrappers use live native Comfy operations."""

from __future__ import annotations

from uuid import UUID

import comfy.ops
import torch
from torch import nn

from simple_syrup.domain.regional_activation_geometry import (
    RegionalActivationBatchAlignment,
    RegionalActivationGeometry,
    RegionalActivationLayout,
)
from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraBranch,
)
from simple_syrup.masking.regional_activation_mask_projection import (
    RegionalActivationMaskBatch,
)
from simple_syrup.runtime.regional_lora.convolution_execution_plan import (
    RegionalConvolutionExecutionPlan,
    RegionalConvolutionParameters,
    RegionalConvolutionTargetUse,
)
from simple_syrup.runtime.regional_lora.convolution_preparation import (
    RegionalConvolutionPreparation,
)
from simple_syrup.runtime.regional_lora.execution_cache import (
    ModelCloneLineage,
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.host_operation_backing import (
    RegionalConvolutionOperationPatch,
    RegionalLinearOperationPatch,
)
from simple_syrup.runtime.regional_lora.linear_execution_plan import (
    RegionalLinearExecutionPlan,
    RegionalLinearOperationKey,
    RegionalLinearTargetUse,
)
from simple_syrup.runtime.regional_lora.operation_invocation import (
    RegionalOperationInvocation,
    RegionalOperationInvocationContext,
    StaticRegionalOperationInvocationResolver,
)
from simple_syrup.runtime.regional_lora.operation_mask_resolution import (
    RegionalOperationMaskBatch,
)
from simple_syrup.runtime.regional_lora.preparation import (
    RegionalLoraTargetPreparation,
)
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget


def test_linear_inactive_path_matches_exact_original_and_preserves_source() -> None:
    """Delegate to the exact native host without registering or copying it."""

    original = comfy.ops.disable_weight_init.Linear(2, 2, bias=False)
    original.weight = nn.Parameter(torch.eye(2), requires_grad=False)
    original.register_buffer("calibration", torch.tensor([3.0]))
    original.eval()
    context = RegionalOperationInvocationContext()
    patch = RegionalLinearOperationPatch(
        "diffusion_model.linear",
        original,
        _linear_plan(),
        context=context,
    )
    inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    torch.testing.assert_close(patch(inputs), original(inputs))
    assert patch.original is original
    assert tuple(patch.parameters()) == ()
    assert tuple(patch.buffers()) == ()
    patch.to(dtype=torch.float64)
    assert original.weight.dtype is torch.float32
    assert original.calibration.dtype is torch.float32


def test_linear_active_path_applies_only_task_local_regional_delta() -> None:
    """Use the installed plan only inside its exact target invocation scope."""

    original = comfy.ops.disable_weight_init.Linear(2, 2, bias=False)
    original.weight = nn.Parameter(torch.eye(2), requires_grad=False)
    context = RegionalOperationInvocationContext()
    patch = RegionalLinearOperationPatch(
        "diffusion_model.linear",
        original,
        _linear_plan(),
        context=context,
    )
    inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    masks = _linear_masks(inputs.shape, torch.tensor([[[[1.0], [0.0]]]]))

    with context.activate(
        StaticRegionalOperationInvocationResolver(
            {"diffusion_model.linear": RegionalOperationInvocation(masks, (1.0,))}
        )
    ):
        output = patch(inputs)

    expected = inputs.clone()
    expected[:, 0] += inputs[:, 0]
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(patch(inputs), inputs)


def test_host_weight_functions_remain_owned_by_live_native_operation() -> None:
    """Use Comfy cast and weight-function state directly from the native host."""

    original = comfy.ops.disable_weight_init.Linear(2, 2, bias=False)
    original.weight = nn.Parameter(torch.eye(2), requires_grad=False)
    patch = RegionalLinearOperationPatch(
        "diffusion_model.linear",
        original,
        _linear_plan(),
    )
    inputs = torch.tensor([[1.0, 2.0]])
    original.weight_function = [lambda weight: weight * 2.0]
    original.bias_function = []
    original.comfy_cast_weights = True

    torch.testing.assert_close(patch(inputs), inputs * 2.0)
    assert patch.original is original


def test_convolution_inactive_and_active_paths_preserve_host_semantics() -> None:
    """Retain Conv2d parameters and add one direct regional convolution delta."""

    original = comfy.ops.disable_weight_init.Conv2d(
        1,
        1,
        1,
        bias=False,
    )
    original.weight = nn.Parameter(torch.ones((1, 1, 1, 1)), requires_grad=False)
    context = RegionalOperationInvocationContext()
    patch = RegionalConvolutionOperationPatch(
        "diffusion_model.conv",
        original,
        _convolution_plan(),
        context=context,
    )
    inputs = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    geometry = RegionalActivationGeometry(
        RegionalActivationLayout.DIRECT_CONVOLUTION_2D,
        (1, 1, 2, 2),
        1,
        2,
        2,
        RegionalActivationBatchAlignment(1, 1),
    )
    spatial_masks = RegionalActivationMaskBatch(
        torch.tensor([[[[[1.0, 0.0], [0.0, 0.0]]]]]),
        geometry,
    )
    masks = RegionalOperationMaskBatch(
        spatial_masks.multipliers,
        spatial_masks.geometry,
        (0,),
    )

    torch.testing.assert_close(patch(inputs), inputs)
    with context.activate(
        StaticRegionalOperationInvocationResolver(
            {"diffusion_model.conv": RegionalOperationInvocation(masks, (1.0,))}
        )
    ):
        output = patch(inputs)

    expected = inputs.clone()
    expected[:, :, 0, 0] *= 2.0
    torch.testing.assert_close(output, expected)
    assert patch.original is original
    assert tuple(patch.parameters()) == ()


def _linear_plan() -> RegionalLinearExecutionPlan:
    """Return one identity ordinary-LoRA plan for a two-feature Linear."""

    identity = RegionalLoraAdapterIdentity("identity.safetensors")
    target = StandardLoraTarget(
        "diffusion_model.linear.weight",
        torch.eye(2),
        torch.eye(2),
        2,
        2,
        2,
        1.0,
    )
    preparation = RegionalLoraTargetPreparation(
        identity,
        ModelCloneLineage(_uuid(1), _uuid(2)),
        target,
        RegionalLoraExecutionCache(),
    )
    return RegionalLinearExecutionPlan(
        (
            RegionalLinearTargetUse(
                0,
                0,
                RegionalLoraBranch.POSITIVE,
                RegionalLinearOperationKey(
                    identity,
                    target.target,
                    id(target.down),
                    id(target.up),
                ),
                preparation,
                1.0,
            ),
        )
    )


def _convolution_plan() -> RegionalConvolutionExecutionPlan:
    """Return one identity 1x1 Conv2d ordinary-LoRA plan."""

    parameters = RegionalConvolutionParameters(
        2,
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        1,
        1,
        (1, 1),
    )
    preparation = RegionalConvolutionPreparation(
        torch.ones((1, 1, 1, 1)),
        None,
        torch.ones((1, 1, 1, 1)),
    )
    return RegionalConvolutionExecutionPlan(
        (
            RegionalConvolutionTargetUse(
                0,
                0,
                RegionalLoraBranch.POSITIVE,
                (id(preparation.down), id(preparation.up), None),
                preparation,
                parameters,
                1.0,
            ),
        )
    )


def _linear_masks(
    shape: torch.Size,
    multipliers: torch.Tensor,
) -> RegionalOperationMaskBatch:
    """Return consumer-spatialized masks for one B/S/C input."""

    geometry = RegionalActivationGeometry(
        RegionalActivationLayout.CONSUMER_SPATIALIZED,
        tuple(shape),
        2,
        1,
        int(shape[1]),
        RegionalActivationBatchAlignment(int(shape[0]), 1),
    )
    spatial = RegionalActivationMaskBatch(multipliers, geometry)
    return RegionalOperationMaskBatch(
        spatial.multipliers,
        spatial.geometry,
        (0,),
    )


def _uuid(value: int) -> UUID:
    """Return one deterministic UUID without leaking construction into fixtures."""

    return UUID(int=value)
