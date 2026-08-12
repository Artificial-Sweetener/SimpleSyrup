# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve reusable exact nonzero support for regional LoRA multipliers."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

import torch

_MAX_CACHED_SIGNATURES = 32


@dataclass(frozen=True, slots=True)
class RegionalLoraActiveSupport:
    """Retain union indices and multiplier values at exactly active positions."""

    indices: torch.Tensor
    multiplier_values: tuple[torch.Tensor, ...]
    element_count: int

    def __post_init__(self) -> None:
        """Require one aligned immutable active-position contract."""

        if self.indices.ndim != 1 or self.indices.dtype is not torch.int64:
            raise ValueError("Regional LoRA active indices must be an int64 vector.")
        if not self.multiplier_values:
            raise ValueError("Regional LoRA active support requires multipliers.")
        if any(
            value.ndim != 1
            or value.shape != self.indices.shape
            or value.device != self.indices.device
            for value in self.multiplier_values
        ):
            raise ValueError("Regional LoRA active multiplier values are misaligned.")


@dataclass(frozen=True, slots=True)
class _RegionalLoraActiveSupportCacheEntry:
    """Retain source tensor identities so allocator reuse cannot alias a cache key."""

    multipliers: tuple[torch.Tensor, ...]
    support: RegionalLoraActiveSupport | None


class RegionalLoraActiveSupportResolver:
    """Own task-local support reuse and dense-execution admission."""

    def __init__(self) -> None:
        """Create one isolated bounded signature cache per execution task."""

        self._cache: ContextVar[
            dict[tuple[object, ...], _RegionalLoraActiveSupportCacheEntry] | None
        ] = ContextVar("simple_syrup_regional_lora_active_support", default=None)

    def resolve(
        self,
        multipliers: tuple[torch.Tensor, ...],
        *,
        leading_shape: tuple[int, ...],
    ) -> RegionalLoraActiveSupport | None:
        """Return sparse exact support or admit the ordinary dense path."""

        if not multipliers:
            raise ValueError("Regional LoRA active support requires multipliers.")
        element_count = 1
        for size in leading_shape:
            element_count *= size
        signature = (
            leading_shape,
            *(
                (
                    id(multiplier),
                    tuple(multiplier.shape),
                    tuple(multiplier.stride()),
                    multiplier.device,
                    multiplier.dtype,
                )
                for multiplier in multipliers
            ),
        )
        cache = self._cache.get()
        if cache is None:
            cache = {}
            self._cache.set(cache)
        if signature in cache:
            entry = cache[signature]
            if all(
                observed is retained
                for observed, retained in zip(
                    multipliers,
                    entry.multipliers,
                    strict=True,
                )
            ):
                return entry.support
            del cache[signature]
        if len(cache) >= _MAX_CACHED_SIGNATURES:
            cache.clear()
        flattened = tuple(
            multiplier.expand(leading_shape).reshape(-1) for multiplier in multipliers
        )
        union = torch.stack(flattened).ne(0).any(dim=0)
        indices = torch.nonzero(union, as_tuple=False).flatten()
        active_count = int(indices.shape[0])
        if active_count == element_count:
            cache[signature] = _RegionalLoraActiveSupportCacheEntry(
                multipliers,
                None,
            )
            return None
        support = RegionalLoraActiveSupport(
            indices,
            tuple(value.index_select(0, indices) for value in flattened),
            element_count,
        )
        cache[signature] = _RegionalLoraActiveSupportCacheEntry(
            multipliers,
            support,
        )
        return support

    def clear(self) -> None:
        """Release the current task's retained multiplier and support tensors."""

        self._cache.set(None)


REGIONAL_LORA_ACTIVE_SUPPORT_RESOLVER = RegionalLoraActiveSupportResolver()
