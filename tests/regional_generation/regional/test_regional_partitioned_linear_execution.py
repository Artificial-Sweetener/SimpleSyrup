# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify base-once compact regional Linear execution semantics."""

from __future__ import annotations

from typing import cast
from unittest.mock import patch
from uuid import UUID

import comfy.ops
import torch
from torch import nn
from torch.nn import functional

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraBranch,
)
from simple_syrup.runtime.regional_lora.active_support import (
    RegionalLoraActiveSupportResolver,
)
from simple_syrup.runtime.regional_lora.execution_cache import (
    ModelCloneLineage,
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.host_linear_parameters import (
    RegionalHostLinearParameterProvider,
)
from simple_syrup.runtime.regional_lora.linear_execution_plan import (
    RegionalLinearExecutionPlan,
    RegionalLinearOperationKey,
    RegionalLinearTargetUse,
)
from simple_syrup.runtime.regional_lora.linear_performance_diagnostics import (
    RegionalLinearPerformanceDiagnostics,
)
from simple_syrup.runtime.regional_lora.partitioned_linear_execution import (
    RegionalPartitionedLinearExecutor,
)
from simple_syrup.runtime.regional_lora.preparation import (
    RegionalLoraTargetPreparation,
)
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget


def test_sparse_duplicate_targets_project_once_and_accumulate_in_order() -> None:
    """Run one base projection and reuse exact A/B deltas across sparse owners."""

    inputs = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4) / 8.0
    module = _module()
    first = _target("diffusion.first", offset=0.0)
    second = _target("diffusion.second", offset=0.25)
    plan = _plan((first, second, first, second))
    multipliers = (
        torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.25, 0.0, 0.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 0.75, 0.0, 1.0]]),
    )
    original_linear = functional.linear
    with patch(
        "simple_syrup.runtime.regional_lora.partitioned_linear_execution.functional.linear",
        wraps=original_linear,
    ) as counted_linear:
        result = _executor().execute(
            inputs,
            plan=plan,
            multipliers=multipliers,
            parameter_provider=RegionalHostLinearParameterProvider(module),
        )

    assert result is not None
    expected = original_linear(inputs, module.weight, module.bias)
    first_delta = original_linear(original_linear(inputs, first.down), first.up)
    second_delta = original_linear(original_linear(inputs, second.down), second.up)
    expected = expected + first_delta * multipliers[0].unsqueeze(-1)
    expected = expected + second_delta * multipliers[1].unsqueeze(-1)
    expected = expected + first_delta * multipliers[2].unsqueeze(-1)
    expected = expected + second_delta * multipliers[3].unsqueeze(-1)
    torch.testing.assert_close(result, expected)
    assert counted_linear.call_count == 5


def test_dense_support_and_zero_groups_preserve_exact_host_base() -> None:
    """Support dense deltas and return the exact base when no group is active."""

    inputs = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4) / 8.0
    module = _module()
    first = _target("diffusion.first", offset=0.0)
    second = _target("diffusion.second", offset=0.25)
    plan = _plan((first, second, first, second))
    executor = _executor()
    provider = RegionalHostLinearParameterProvider(module)

    dense = executor.execute(
        inputs,
        plan=plan,
        multipliers=(torch.ones((1, 6)), None, None, None),
        parameter_provider=provider,
    )
    base = functional.linear(inputs, module.weight, module.bias)
    delta = functional.linear(functional.linear(inputs, first.down), first.up)
    assert dense is not None
    torch.testing.assert_close(dense, base + delta)

    zero = executor.execute(
        inputs,
        plan=plan,
        multipliers=(None, None, None, None),
        parameter_provider=provider,
    )
    assert zero is not None
    torch.testing.assert_close(zero, base)


def test_missing_host_provider_declines_without_projection() -> None:
    """Leave unsupported host calls to the authoritative ordered executor."""

    target = _target("diffusion.target", offset=0.0)

    result = _executor().execute(
        torch.ones((1, 2, 4)),
        plan=_plan((target, target)),
        multipliers=(torch.ones((1, 2)), torch.ones((1, 2))),
        parameter_provider=None,
    )

    assert result is None


def test_plain_linear_uses_direct_parameters_without_comfy_cast() -> None:
    """Acquire an ordinary host Linear through its native parameter contract."""

    module = nn.Linear(4, 3, bias=True)
    provider = RegionalHostLinearParameterProvider(module)

    with patch("comfy.ops.cast_bias_weight") as comfy_cast:
        with provider.acquire(torch.ones((1, 2, 4))) as parameters:
            assert parameters.weight is module.weight
            assert parameters.bias is module.bias

    comfy_cast.assert_not_called()


def _executor() -> RegionalPartitionedLinearExecutor:
    """Return one isolated executor with no cross-test support retention."""

    return RegionalPartitionedLinearExecutor(
        RegionalLoraActiveSupportResolver(),
        RegionalLinearPerformanceDiagnostics(),
    )


def _module() -> nn.Linear:
    """Return one generic Comfy host Linear with explicit stable parameters."""

    module = cast(nn.Linear, comfy.ops.disable_weight_init.Linear(4, 3, bias=True))
    module.weight = nn.Parameter(
        torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10.0,
        requires_grad=False,
    )
    module.bias = nn.Parameter(
        torch.tensor([0.1, 0.2, 0.3]),
        requires_grad=False,
    )
    return module


def _target(name: str, *, offset: float) -> StandardLoraTarget:
    """Return one generic rank-two target."""

    down = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8.0 + offset
    up = torch.arange(6, dtype=torch.float32).reshape(3, 2) / 6.0 + offset
    return StandardLoraTarget(name, down, up, 2, 4, 3, 1.0)


def _plan(targets: tuple[StandardLoraTarget, ...]) -> RegionalLinearExecutionPlan:
    """Return ordered groups while preserving repeated noncontiguous targets."""

    cache = RegionalLoraExecutionCache()
    lineage = ModelCloneLineage(UUID(int=1), UUID(int=2))
    uses: list[RegionalLinearTargetUse] = []
    for index, target in enumerate(targets):
        identity = RegionalLoraAdapterIdentity(f"adapter-{index % 2}")
        uses.append(
            RegionalLinearTargetUse(
                index,
                index % 2,
                (
                    RegionalLoraBranch.POSITIVE
                    if index < 2
                    else RegionalLoraBranch.NEGATIVE
                ),
                RegionalLinearOperationKey(
                    identity,
                    target.target,
                    id(target.down),
                    id(target.up),
                ),
                RegionalLoraTargetPreparation(
                    identity,
                    lineage,
                    target,
                    cache,
                ),
                1.0,
            )
        )
    return RegionalLinearExecutionPlan(tuple(uses))
