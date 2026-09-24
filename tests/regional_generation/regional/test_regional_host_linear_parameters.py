# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove regional Linear parameters follow Comfy's native cast lifecycle."""

from __future__ import annotations

from unittest.mock import patch

import comfy.ops
import pytest
import torch
from torch import nn

from simple_syrup.runtime.regional_lora.host_linear_parameters import (
    RegionalHostLinearParameterProvider,
)


def test_loaded_linear_uses_exact_live_parameters_without_casting() -> None:
    """Reuse resident weights when native Comfy would call ordinary Linear."""

    module = _linear()
    inputs = torch.tensor([[1.0, 2.0]])
    provider = RegionalHostLinearParameterProvider(module)

    with (
        patch.object(comfy.ops, "run_every_op") as run_every_op,
        patch.object(comfy.ops, "cast_bias_weight") as cast_bias_weight,
    ):
        with provider.acquire(inputs) as parameters:
            assert parameters.weight is module.weight
            assert parameters.bias is module.bias

    run_every_op.assert_called_once_with()
    cast_bias_weight.assert_not_called()


@pytest.mark.parametrize("function_attribute", ["weight_function", "bias_function"])
def test_patch_functions_select_native_cast_lifecycle(
    function_attribute: str,
) -> None:
    """Cast when either native Comfy patch-function collection is active."""

    module = _linear()
    setattr(module, function_attribute, [lambda value: value])
    _assert_cast_lifecycle(module)


def test_comfy_cast_flag_selects_native_cast_lifecycle() -> None:
    """Cast when DynamicVRAM marks the native module for cast execution."""

    module = _linear()
    module.comfy_cast_weights = True
    _assert_cast_lifecycle(module)


def test_cast_lifecycle_uncasts_when_consumer_raises() -> None:
    """Release Comfy offload ownership even when regional execution fails."""

    module = _linear()
    module.comfy_cast_weights = True
    inputs = torch.tensor([[1.0, 2.0]])
    cast_weight = torch.full_like(module.weight, 3.0)
    cast_bias = torch.full_like(module.bias, 4.0)
    offload = object()
    provider = RegionalHostLinearParameterProvider(module)

    with (
        patch.object(comfy.ops, "run_every_op"),
        patch.object(
            comfy.ops,
            "cast_bias_weight",
            return_value=(cast_weight, cast_bias, offload),
        ),
        patch.object(comfy.ops, "uncast_bias_weight") as uncast_bias_weight,
        pytest.raises(RuntimeError, match="consumer failure"),
    ):
        with provider.acquire(inputs):
            raise RuntimeError("consumer failure")

    uncast_bias_weight.assert_called_once_with(
        module,
        cast_weight,
        cast_bias,
        offload,
    )


def _assert_cast_lifecycle(module: comfy.ops.disable_weight_init.Linear) -> None:
    """Prove one admitted native cast path yields and releases exact values."""

    inputs = torch.tensor([[1.0, 2.0]])
    weight = module.weight
    bias = module.bias
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise AssertionError("Test Linear parameters are unavailable.")
    cast_weight = torch.full_like(weight, 3.0)
    cast_bias = torch.full_like(bias, 4.0)
    offload = object()
    provider = RegionalHostLinearParameterProvider(module)

    with (
        patch.object(comfy.ops, "run_every_op") as run_every_op,
        patch.object(
            comfy.ops,
            "cast_bias_weight",
            return_value=(cast_weight, cast_bias, offload),
        ) as cast_bias_weight,
        patch.object(comfy.ops, "uncast_bias_weight") as uncast_bias_weight,
    ):
        with provider.acquire(inputs) as parameters:
            assert parameters.weight is cast_weight
            assert parameters.bias is cast_bias

    run_every_op.assert_called_once_with()
    cast_bias_weight.assert_called_once_with(module, inputs, offloadable=True)
    uncast_bias_weight.assert_called_once_with(
        module,
        cast_weight,
        cast_bias,
        offload,
    )


def _linear() -> comfy.ops.disable_weight_init.Linear:
    """Return one resident Comfy Linear with empty patch-function state."""

    module = comfy.ops.disable_weight_init.Linear(2, 2, bias=True)
    module.weight = nn.Parameter(torch.eye(2), requires_grad=False)
    module.bias = nn.Parameter(torch.zeros(2), requires_grad=False)
    module.comfy_cast_weights = False
    module.weight_function = []
    module.bias_function = []
    return module
