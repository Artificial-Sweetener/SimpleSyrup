# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# mypy: disable-error-code="no-untyped-def"
# ruff: noqa: ANN001, ANN202

"""Accumulate ordered adapter outputs with exact execution-dtype rounding."""

from __future__ import annotations

from typing import Any, cast

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]

_MAX_DELTAS_PER_LAUNCH = 8
_BLOCK_SIZE = 256


@triton.jit  # type: ignore[untyped-decorator]
def _ordered_accumulation_kernel(
    output,
    base,
    delta_0,
    delta_1,
    delta_2,
    delta_3,
    delta_4,
    delta_5,
    delta_6,
    delta_7,
    element_count,
    delta_count: tl.constexpr,
    execution_dtype: tl.constexpr,
    block_size: tl.constexpr,
):
    """Add one ordered chunk and round after every declared adapter."""

    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    active = offsets < element_count
    value = tl.load(base + offsets, mask=active)
    if delta_count > 0:
        value = (value + tl.load(delta_0 + offsets, mask=active)).to(execution_dtype)
    if delta_count > 1:
        value = (value + tl.load(delta_1 + offsets, mask=active)).to(execution_dtype)
    if delta_count > 2:
        value = (value + tl.load(delta_2 + offsets, mask=active)).to(execution_dtype)
    if delta_count > 3:
        value = (value + tl.load(delta_3 + offsets, mask=active)).to(execution_dtype)
    if delta_count > 4:
        value = (value + tl.load(delta_4 + offsets, mask=active)).to(execution_dtype)
    if delta_count > 5:
        value = (value + tl.load(delta_5 + offsets, mask=active)).to(execution_dtype)
    if delta_count > 6:
        value = (value + tl.load(delta_6 + offsets, mask=active)).to(execution_dtype)
    if delta_count > 7:
        value = (value + tl.load(delta_7 + offsets, mask=active)).to(execution_dtype)
    tl.store(output + offsets, value, mask=active)


class OrderedTensorAccumulator:
    """Own exact ordered accumulation and its CUDA launch policy."""

    def accumulate(
        self,
        base: torch.Tensor,
        deltas: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Add deltas into a fresh base tensor in declared dtype-rounding order."""

        self._validate(base, deltas)
        if not deltas:
            return base
        if base.device.type != "cuda":
            return self._torch_accumulate(base, deltas)
        execution_dtype = self._triton_dtype(base.dtype)
        remaining = deltas
        result = base
        while remaining:
            chunk = remaining[:_MAX_DELTAS_PER_LAUNCH]
            remaining = remaining[_MAX_DELTAS_PER_LAUNCH:]
            padded = (*chunk, *((result,) * (_MAX_DELTAS_PER_LAUNCH - len(chunk))))
            grid = (triton.cdiv(result.numel(), _BLOCK_SIZE),)
            kernel = cast(Any, _ordered_accumulation_kernel)
            kernel[grid](
                result,
                result,
                *padded,
                result.numel(),
                delta_count=len(chunk),
                execution_dtype=execution_dtype,
                block_size=_BLOCK_SIZE,
            )
        return result

    @staticmethod
    def _torch_accumulate(
        base: torch.Tensor,
        deltas: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Preserve ordinary ordered Torch semantics outside CUDA."""

        result = base
        for delta in deltas:
            result.add_(delta)
        return result

    @staticmethod
    def _triton_dtype(dtype: torch.dtype) -> Any:
        """Map the admitted floating execution dtype to a Triton scalar dtype."""

        if dtype is torch.bfloat16:
            return tl.bfloat16
        if dtype is torch.float16:
            return tl.float16
        if dtype is torch.float32:
            return tl.float32
        raise TypeError(f"Ordered CUDA accumulation does not support {dtype}.")

    @staticmethod
    def _validate(
        base: torch.Tensor,
        deltas: tuple[torch.Tensor, ...],
    ) -> None:
        """Require one contiguous execution contract across all operands."""

        if not isinstance(base, torch.Tensor) or not base.is_floating_point():
            raise TypeError("Ordered accumulation base must be a floating tensor.")
        if base.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError("Ordered accumulation dtype is unsupported.")
        if not base.is_contiguous():
            raise ValueError("Ordered accumulation base must be contiguous.")
        if not isinstance(deltas, tuple):
            raise TypeError("Ordered accumulation deltas must be an immutable tuple.")
        for delta in deltas:
            if not isinstance(delta, torch.Tensor):
                raise TypeError("Ordered accumulation delta must be a tensor.")
            if (
                delta.shape != base.shape
                or delta.device != base.device
                or delta.dtype != base.dtype
                or not delta.is_contiguous()
            ):
                raise ValueError(
                    "Ordered accumulation deltas must match the contiguous base."
                )


ORDERED_TENSOR_ACCUMULATOR = OrderedTensorAccumulator()
