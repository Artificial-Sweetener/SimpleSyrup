# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own task-local standard-UNet attn2 resolution execution caching."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from .unet_attention_resolution_key import StandardUnetAttentionResolutionKey
from .unet_attn2_execution import UnetAttn2Execution


class StandardUnetAttn2ResolutionCache:
    """Cache exact executions only inside one nested diffusion call."""

    def __init__(self) -> None:
        """Create a task-local slot with no process-global active cache."""

        self._current: ContextVar[
            dict[StandardUnetAttentionResolutionKey, UnetAttn2Execution] | None
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
        key: StandardUnetAttentionResolutionKey,
        factory: Callable[[], UnetAttn2Execution],
    ) -> UnetAttn2Execution:
        """Return one cached execution or build and store a validated result."""

        if not isinstance(key, StandardUnetAttentionResolutionKey):
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
