# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Accumulate ordered adapter outputs with exact execution-dtype rounding."""

from __future__ import annotations

from types import ModuleType
from typing import Protocol, cast

import torch

from .triton_runtime import TRITON_RUNTIME_RESOLVER, TritonRuntimeResolver

_TRITON_BACKEND_MODULE = (
    "simple_syrup.runtime.regional_lora.ordered_accumulation_triton"
)


class _OrderedAccumulationBackend(Protocol):
    """Describe the lazy CUDA backend surface consumed by this owner."""

    def accumulate(
        self,
        base: torch.Tensor,
        deltas: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Accumulate validated CUDA tensors in declared order."""

        ...


class OrderedTensorAccumulator:
    """Own exact ordered accumulation and its CUDA launch policy."""

    def __init__(
        self,
        resolver: TritonRuntimeResolver = TRITON_RUNTIME_RESOLVER,
    ) -> None:
        """Retain the process-level optional acceleration authority."""

        if not isinstance(resolver, TritonRuntimeResolver):
            raise TypeError("Ordered accumulation requires a Triton resolver.")
        self._resolver = resolver

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
        backend = self._resolver.resolve(_TRITON_BACKEND_MODULE)
        if backend is None:
            return self._torch_accumulate(base, deltas)
        return cast(_OrderedAccumulationBackend, cast(ModuleType, backend)).accumulate(
            base,
            deltas,
        )

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
                    "Ordered accumulation deltas must match the contiguous base; "
                    f"base(shape={tuple(base.shape)!r}, device={base.device!s}, "
                    f"dtype={base.dtype!s}, contiguous={base.is_contiguous()}) "
                    f"delta(shape={tuple(delta.shape)!r}, device={delta.device!s}, "
                    f"dtype={delta.dtype!s}, contiguous={delta.is_contiguous()})."
                )


ORDERED_TENSOR_ACCUMULATOR = OrderedTensorAccumulator()
