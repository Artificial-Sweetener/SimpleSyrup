# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove exact declared-order tensor accumulation on every execution path."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.runtime.regional_lora.ordered_accumulation import (
    OrderedTensorAccumulator,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cpu_accumulation_matches_stepwise_execution_dtype_rounding(
    dtype: torch.dtype,
) -> None:
    """Match ordinary ordered Torch additions and mutate the fresh base output."""

    base = torch.tensor([[1.0, 64.0, -0.5]], dtype=dtype)
    deltas = tuple(
        torch.tensor([[value, value / 8, -value]], dtype=dtype)
        for value in (0.25, 0.03125, -0.125, 0.0078125)
    )
    expected = _reference(base.clone(), deltas)

    result = OrderedTensorAccumulator().accumulate(base, deltas)

    assert torch.equal(result, expected)
    assert result.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("count", [2, 4, 8, 11])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_accumulation_matches_every_intermediate_rounding(
    count: int,
    dtype: torch.dtype,
) -> None:
    """Match stepwise Torch results across one and multiple Triton launches."""

    generator = torch.Generator(device="cuda").manual_seed(72_941)
    base = torch.randn((3, 17, 29), generator=generator, device="cuda", dtype=dtype)
    deltas = tuple(
        torch.randn(
            base.shape,
            generator=generator,
            device=base.device,
            dtype=dtype,
        )
        for _ in range(count)
    )
    expected = _reference(base.clone(), deltas)

    result = OrderedTensorAccumulator().accumulate(base, deltas)
    torch.cuda.synchronize()

    assert torch.equal(result, expected)


def _reference(
    base: torch.Tensor,
    deltas: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Build the explicit eager addition-order authority."""

    result = base
    for delta in deltas:
        result = result + delta
    return result
