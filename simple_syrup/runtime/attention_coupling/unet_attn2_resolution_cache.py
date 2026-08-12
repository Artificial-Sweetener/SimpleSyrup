# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own task-local standard-UNet attn2 resolution execution caching."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from ...domain.spatial_views import SpatialBatchLayout
from .unet_attn2_execution import UnetAttn2Execution
from .unet_attn2_geometry import StandardUnetAttn2Geometry


@dataclass(frozen=True, slots=True, eq=False)
class StandardUnetAttn2ResolutionKey:
    """Identify every call-local value that changes projected execution."""

    input_batch_size: int
    query_height: int
    query_width: int
    original_height: int
    original_width: int
    device: torch.device
    dtype: torch.dtype
    published_layout: SpatialBatchLayout | None

    @classmethod
    def from_geometry(
        cls,
        geometry: StandardUnetAttn2Geometry,
        query: torch.Tensor,
    ) -> StandardUnetAttn2ResolutionKey:
        """Build a key from validated geometry and query residency."""

        if not isinstance(geometry, StandardUnetAttn2Geometry):
            raise TypeError("UNet resolution key requires validated geometry.")
        if not isinstance(query, torch.Tensor):
            raise TypeError("UNet resolution key requires a query tensor.")
        return cls(
            geometry.query.input_batch_size,
            geometry.query.query_height,
            geometry.query.query_width,
            geometry.original_height,
            geometry.original_width,
            query.device,
            query.dtype,
            geometry.query.spatial_layout,
        )

    def __hash__(self) -> int:
        """Hash exact scalar residency plus published layout identity."""

        return hash(
            (
                self.input_batch_size,
                self.query_height,
                self.query_width,
                self.original_height,
                self.original_width,
                self.device,
                self.dtype,
                id(self.published_layout),
            )
        )

    def __eq__(self, other: object) -> bool:
        """Compare every scalar and require exact published layout identity."""

        return (
            isinstance(other, StandardUnetAttn2ResolutionKey)
            and self.input_batch_size == other.input_batch_size
            and self.query_height == other.query_height
            and self.query_width == other.query_width
            and self.original_height == other.original_height
            and self.original_width == other.original_width
            and self.device == other.device
            and self.dtype == other.dtype
            and self.published_layout is other.published_layout
        )


class StandardUnetAttn2ResolutionCache:
    """Cache exact executions only inside one nested diffusion call."""

    def __init__(self) -> None:
        """Create a task-local slot with no process-global active cache."""

        self._current: ContextVar[
            dict[StandardUnetAttn2ResolutionKey, UnetAttn2Execution] | None
        ] = ContextVar(
            "simple_syrup_unet_attn2_resolution_cache",
            default=None,
        )

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Publish one empty cache for exactly one nested diffusion call."""

        token = self._current.set({})
        try:
            yield
        finally:
            self._current.reset(token)

    def resolve(
        self,
        key: StandardUnetAttn2ResolutionKey,
        factory: Callable[[], UnetAttn2Execution],
    ) -> UnetAttn2Execution:
        """Return one cached execution or build and store a validated result."""

        if not isinstance(key, StandardUnetAttn2ResolutionKey):
            raise TypeError("UNet resolution cache requires a typed key.")
        cache = self._current.get()
        if cache is None:
            raise RuntimeError(
                "UNet resolution cache is unavailable outside its diffusion wrapper."
            )
        existing = cache.get(key)
        if existing is not None:
            return existing
        execution = factory()
        if not isinstance(execution, UnetAttn2Execution):
            raise TypeError("UNet resolution factory returned an invalid execution.")
        cache[key] = execution
        return execution

    @property
    def size(self) -> int:
        """Return current call-local cache size or reject outside its scope."""

        cache = self._current.get()
        if cache is None:
            raise RuntimeError(
                "UNet resolution cache is unavailable outside its diffusion wrapper."
            )
        return len(cache)
