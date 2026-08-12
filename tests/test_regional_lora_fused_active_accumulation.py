# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove fused CUDA regional-LoRA B projection and ordered accumulation."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from simple_syrup.runtime.regional_lora.fused_active_accumulation import (
    RegionalLoraFusedActiveAccumulator,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Fused regional LoRA accumulation requires CUDA.",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("adapter_count", [1, 4, 9])
@pytest.mark.parametrize("indexed", [False, True])
def test_fused_accumulation_matches_ordered_torch_reference(
    dtype: torch.dtype,
    adapter_count: int,
    indexed: bool,
) -> None:
    """Match dense/indexed rows and chunked adapter order at native precision."""

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(
        71 + adapter_count + int(indexed)
    )
    output_rows = 17
    output_features = 70
    rank = 32
    indices = (
        torch.tensor([0, 2, 3, 6, 8, 11, 13, 15, 16], device=device)
        if indexed
        else None
    )
    active_rows = output_rows if indices is None else int(indices.shape[0])
    output = torch.randn(
        (output_rows, output_features),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    rank_values = (
        torch.randn(
            (active_rows, adapter_count, rank),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        * 0.05
    )
    up = (
        torch.randn(
            (adapter_count, output_features, rank),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        * 0.05
    )
    multipliers = tuple(
        torch.randn(
            (active_rows,),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        for _index in range(adapter_count)
    )
    expected = _torch_reference(
        output,
        rank_values=rank_values,
        up=up,
        multipliers=multipliers,
        indices=indices,
    )
    output_pointer = output.untyped_storage().data_ptr()

    result = RegionalLoraFusedActiveAccumulator().add(
        output,
        rank_values=rank_values,
        up=up,
        multipliers=multipliers,
        indices=indices,
    )

    assert result.untyped_storage().data_ptr() == output_pointer
    tolerance = 0.002 if dtype is torch.float16 else 0.02
    torch.testing.assert_close(result, expected, rtol=tolerance, atol=tolerance)
    if indices is not None:
        inactive = torch.ones(output_rows, device=device, dtype=torch.bool)
        inactive[indices] = False
        torch.testing.assert_close(result[inactive], expected[inactive], rtol=0, atol=0)


def test_fused_accumulation_admits_only_verified_cuda_execution_types() -> None:
    """Leave CPU and float32 execution on the established Torch path."""

    accumulator = RegionalLoraFusedActiveAccumulator()

    assert not accumulator.supports(torch.zeros((2, 2)), torch.zeros((1, 2, 1)))
    assert not accumulator.supports(
        torch.zeros((2, 2), device="cuda", dtype=torch.float32),
        torch.zeros((1, 2, 1), device="cuda", dtype=torch.float32),
    )
    assert accumulator.supports(
        torch.zeros((2, 2), device="cuda", dtype=torch.bfloat16),
        torch.zeros((1, 2, 1), device="cuda", dtype=torch.bfloat16),
    )


def test_single_adapter_reuses_multiplier_without_stack_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the singleton hot path free of a redundant device copy launch."""

    device = torch.device("cuda")
    output = torch.zeros((2, 16), device=device, dtype=torch.bfloat16)
    rank_values = torch.ones((2, 1, 16), device=device, dtype=torch.bfloat16)
    up = torch.ones((1, 16, 16), device=device, dtype=torch.bfloat16)
    multiplier = torch.ones((2,), device=device, dtype=torch.bfloat16)

    def fail_stack(*values: object, **options: object) -> torch.Tensor:
        """Fail if singleton accumulation allocates a stacked multiplier."""

        raise AssertionError("single-adapter accumulation must not call torch.stack")

    monkeypatch.setattr(torch, "stack", fail_stack)

    result = RegionalLoraFusedActiveAccumulator().add(
        output,
        rank_values=rank_values,
        up=up,
        multipliers=(multiplier,),
        indices=None,
    )

    torch.testing.assert_close(result, torch.full_like(result, 16.0))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("indexed", [False, True])
def test_target_selection_reads_shared_rank_storage_without_slicing(
    dtype: torch.dtype,
    indexed: bool,
) -> None:
    """Match one selected target from a larger compatible rank batch."""

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(404 + int(indexed))
    output_rows = 19
    output_features = 67
    target_count = 3
    target_index = 1
    rank = 32
    indices = (
        torch.tensor([0, 2, 4, 7, 8, 11, 14, 16, 18], device=device)
        if indexed
        else None
    )
    active_rows = output_rows if indices is None else int(indices.shape[0])
    output = torch.randn(
        (output_rows, output_features),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    rank_values = torch.randn(
        (active_rows, target_count, rank),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    up = torch.randn(
        (target_count, output_features, rank),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    multiplier = torch.randn(
        (active_rows,),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    expected = output.clone()
    active_output = expected if indices is None else expected.index_select(0, indices)
    active_output.add_(
        functional.linear(
            rank_values[:, target_index] * multiplier[:, None],
            up[target_index],
        )
    )
    if indices is not None:
        expected.index_copy_(0, indices, active_output)
    rank_pointer = rank_values.untyped_storage().data_ptr()
    up_pointer = up.untyped_storage().data_ptr()

    result = RegionalLoraFusedActiveAccumulator().add_target(
        output,
        rank_values=rank_values,
        up=up,
        multiplier=multiplier,
        indices=indices,
        target_index=target_index,
    )

    tolerance = 0.002 if dtype is torch.float16 else 0.02
    torch.testing.assert_close(result, expected, rtol=tolerance, atol=tolerance)
    assert rank_values.untyped_storage().data_ptr() == rank_pointer
    assert up.untyped_storage().data_ptr() == up_pointer


def _torch_reference(
    output: torch.Tensor,
    *,
    rank_values: torch.Tensor,
    up: torch.Tensor,
    multipliers: tuple[torch.Tensor, ...],
    indices: torch.Tensor | None,
) -> torch.Tensor:
    """Evaluate the superseded separate B/add/scatter path exactly."""

    result = output.clone()
    active = result if indices is None else result.index_select(0, indices)
    for adapter_index, multiplier in enumerate(multipliers):
        delta = functional.linear(
            rank_values[:, adapter_index] * multiplier[:, None],
            up[adapter_index],
        )
        active.add_(delta)
    if indices is not None:
        result.index_copy_(0, indices, active)
    return result
