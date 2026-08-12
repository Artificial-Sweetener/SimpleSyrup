# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove exact full-rank regional LoRA preparation and delta math."""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest
import torch

from simple_syrup.domain.regional_lora_plan import RegionalLoraAdapterIdentity
from simple_syrup.runtime.regional_lora.delta_execution import RegionalLoraDeltaExecutor
from simple_syrup.runtime.regional_lora.execution_cache import (
    ModelCloneLineage,
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.preparation import (
    RegionalLoraCompatibleBatchPreparation,
    RegionalLoraTargetPreparation,
)
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget


class _ExplodingLock:
    """Fail if an already-prepared hot cache hit re-enters its miss lock."""

    def __enter__(self) -> None:
        """Reject lock entry on the prepared fast path."""

        raise AssertionError("prepared cache hit re-entered its miss lock")

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        """Provide the context-manager shape for the deliberately failing lock."""

        del exception_type, exception, traceback


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_delta_matches_exact_two_linear_formula_at_full_rank(
    dtype: torch.dtype,
) -> None:
    """Match strength times B(A(x)) without rank reduction in each runtime dtype."""

    down = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.5, 1.0]])
    up = torch.tensor([[2.0, -1.0], [0.25, 3.0]])
    target = StandardLoraTarget("target", down, up, 2, 3, 2)
    preparation = _preparation(target)
    inputs = torch.tensor(
        [[[1.0, 0.5, -2.0], [0.25, -1.0, 3.0]]],
        dtype=dtype,
    )

    delta = RegionalLoraDeltaExecutor().delta(
        inputs,
        preparation=preparation,
        strength=-0.75,
    )

    expected = ((inputs @ down.to(dtype).T) @ up.to(dtype).T) * -0.75
    tolerance = 1e-6 if dtype is torch.float32 else 2e-2
    torch.testing.assert_close(delta, expected, atol=tolerance, rtol=tolerance)
    assert delta.dtype is dtype
    assert preparation.weights(device=inputs.device, dtype=dtype).down.shape[0] == 2


def test_preparation_reuses_cache_without_aliasing_or_mutating_cpu_sources() -> None:
    """Prepare one device/dtype pair once while retaining source bytes and identity."""

    down = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    up = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    target = StandardLoraTarget("target", down, up, 2, 3, 2)
    cache = RegionalLoraExecutionCache()
    preparation = _preparation(target, cache=cache)
    down_before = down.clone()
    up_before = up.clone()

    first = preparation.weights(device=torch.device("cpu"), dtype=torch.float32)
    second = preparation.weights(device=torch.device("cpu"), dtype=torch.float32)

    assert first is second
    assert cache.size == 1
    assert first.down is not down
    assert first.up is not up
    assert torch.equal(down, down_before)
    assert torch.equal(up, up_before)


def test_prepared_weight_hot_hit_does_not_reenter_miss_lock() -> None:
    """Keep every warmed target lookup on the lock-free immutable read path."""

    target = StandardLoraTarget(
        "target",
        torch.ones((2, 3)),
        torch.ones((2, 2)),
        2,
        3,
        2,
    )
    preparation = _preparation(target)
    expected = preparation.weights(device=torch.device("cpu"), dtype=torch.float32)
    cast(Any, preparation)._lock = _ExplodingLock()

    observed = preparation.weights(device=torch.device("cpu"), dtype=torch.float32)

    assert observed is expected


def test_masked_delta_matches_rank_space_reference_through_two_linears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the direct singleton path without stack or einsum machinery."""

    target = StandardLoraTarget(
        "target",
        torch.tensor([[1.0, 0.5], [-0.5, 1.0]]),
        torch.tensor([[1.0, 0.25], [0.5, -1.0]]),
        2,
        2,
        2,
    )
    inputs = torch.tensor([[[1.0, -0.5], [0.25, 2.0]]])
    multiplier = torch.tensor([[1.0, 0.25]])
    original_linear = torch.nn.functional.linear
    linear_calls = 0

    def counted_linear(
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Count both exact singleton projections."""

        nonlocal linear_calls
        linear_calls += 1
        return original_linear(inputs, weight, bias)

    monkeypatch.setattr(
        "simple_syrup.runtime.regional_lora.single_active_projection.functional.linear",
        counted_linear,
    )

    output = RegionalLoraDeltaExecutor().masked_delta(
        inputs,
        preparation=_preparation(target),
        multiplier=multiplier,
    )

    expected = ((inputs @ target.down.T) * multiplier.unsqueeze(-1)) @ target.up.T
    torch.testing.assert_close(output, expected)
    assert linear_calls == 2


def test_add_masked_delta_fuses_base_accumulation_into_up_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match ordered base-plus-delta math with one down linear and one addmm."""

    target = StandardLoraTarget(
        "target",
        torch.tensor([[1.0, 0.5], [-0.5, 1.0]]),
        torch.tensor([[1.0, 0.25], [0.5, -1.0]]),
        2,
        2,
        2,
    )
    inputs = torch.tensor([[[1.0, -0.5], [0.25, 2.0]]])
    original_output = torch.tensor([[[3.0, -2.0], [0.5, 1.25]]])
    multiplier = torch.tensor([[1.0, 0.25]])
    original_before = original_output.clone()
    original_addmm = torch.Tensor.addmm_
    addmm_calls = 0

    def counted_addmm(
        tensor: torch.Tensor,
        mat1: torch.Tensor,
        mat2: torch.Tensor,
    ) -> torch.Tensor:
        """Count the fused accumulation while preserving torch behavior."""

        nonlocal addmm_calls
        addmm_calls += 1
        return original_addmm(tensor, mat1, mat2)

    monkeypatch.setattr(torch.Tensor, "addmm_", counted_addmm)

    output = RegionalLoraDeltaExecutor().add_masked_delta(
        original_output,
        inputs,
        preparation=_preparation(target),
        multiplier=multiplier,
    )

    expected_delta = ((inputs @ target.down.T) * multiplier.unsqueeze(-1)) @ target.up.T
    torch.testing.assert_close(output, original_before + expected_delta)
    assert (
        output.untyped_storage().data_ptr()
        == original_output.untyped_storage().data_ptr()
    )
    assert addmm_calls == 1


def test_compatible_batch_applies_each_multiplier_in_rank_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match independently masked B(A(x)) values in grouped batch projections."""

    first = StandardLoraTarget(
        "target",
        torch.tensor([[1.0, 0.5], [-0.5, 1.0]]),
        torch.tensor([[1.0, 0.25], [0.5, -1.0]]),
        2,
        2,
        2,
    )
    second = StandardLoraTarget(
        "target",
        torch.tensor([[0.25, 1.0], [1.5, -0.5]]),
        torch.tensor([[0.75, -0.25], [1.0, 0.5]]),
        2,
        2,
        2,
    )
    inputs = torch.tensor([[[1.0, -0.5], [0.25, 2.0]]])
    multipliers = (
        torch.tensor([[1.0, 0.25]]),
        torch.tensor([[0.5, -0.75]]),
    )
    preparation = RegionalLoraCompatibleBatchPreparation(
        (_preparation(first), _preparation(second))
    )
    original_linear = torch.nn.functional.linear
    linear_calls = 0

    def counted_linear(
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Count the shared A projection while preserving functional behavior."""

        nonlocal linear_calls
        linear_calls += 1
        return original_linear(inputs, weight, bias)

    monkeypatch.setattr(
        "simple_syrup.runtime.regional_lora.compatible_rank_projection.functional.linear",
        counted_linear,
    )

    outputs = RegionalLoraDeltaExecutor().compatible_batch(
        inputs,
        preparation=preparation,
        multipliers=multipliers,
    )

    expected = torch.stack(
        (
            ((inputs @ first.down.T) * multipliers[0].unsqueeze(-1)) @ first.up.T,
            ((inputs @ second.down.T) * multipliers[1].unsqueeze(-1)) @ second.up.T,
        ),
        dim=-2,
    )
    torch.testing.assert_close(outputs, expected)
    assert linear_calls == 1
    assert len(preparation._prepared) == 1


def test_compatible_batch_rejects_different_ranks_or_shapes() -> None:
    """Keep incompatible targets out of one batched low-rank projection."""

    first = StandardLoraTarget(
        "target",
        torch.ones((1, 2)),
        torch.ones((2, 1)),
        1,
        2,
        2,
    )
    second = StandardLoraTarget(
        "target",
        torch.ones((2, 2)),
        torch.ones((2, 2)),
        2,
        2,
        2,
    )

    with pytest.raises(ValueError, match="share rank and shape"):
        RegionalLoraCompatibleBatchPreparation(
            (_preparation(first), _preparation(second))
        )


@pytest.mark.parametrize(
    ("inputs", "strength", "error", "message"),
    [
        (torch.ones((1, 4)), 1.0, ValueError, "feature dimension"),
        (torch.ones((1, 3), dtype=torch.int64), 1.0, TypeError, "floating tensor"),
        (torch.ones((1, 3)), float("nan"), TypeError, "finite numeric"),
    ],
)
def test_delta_rejects_invalid_inputs_before_matrix_work(
    inputs: torch.Tensor,
    strength: float,
    error: type[Exception],
    message: str,
) -> None:
    """Fail closed on feature, dtype, and strength contract violations."""

    target = StandardLoraTarget(
        "target",
        torch.ones((1, 3)),
        torch.ones((2, 1)),
        1,
        3,
        2,
    )

    with pytest.raises(error, match=message):
        RegionalLoraDeltaExecutor().delta(
            inputs,
            preparation=_preparation(target),
            strength=strength,
        )


def _preparation(
    target: StandardLoraTarget,
    *,
    cache: RegionalLoraExecutionCache | None = None,
) -> RegionalLoraTargetPreparation:
    """Build one preparation with stable synthetic Comfy lineage."""

    return RegionalLoraTargetPreparation(
        adapter_identity=RegionalLoraAdapterIdentity("adapter.safetensors"),
        model_lineage=ModelCloneLineage(uuid4(), uuid4()),
        target=target,
        cache=cache or RegionalLoraExecutionCache(),
    )
