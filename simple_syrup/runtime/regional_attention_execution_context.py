# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Publish model-call-scoped regional attention context batches."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from ..domain.regional_attention_batch import BatchedRegionalAttentionContexts


class RegionalAttentionExecutionContext:
    """Own task-local dynamic context alignment for one patched MODEL."""

    def __init__(self) -> None:
        """Create an empty model-call-scoped context slot."""

        self._current: ContextVar[BatchedRegionalAttentionContexts | None] = ContextVar(
            "simple_syrup_regional_attention_contexts",
            default=None,
        )

    def current_or(
        self,
        fallback: BatchedRegionalAttentionContexts,
    ) -> BatchedRegionalAttentionContexts:
        """Return active model-call contexts or the immutable static fallback."""

        if not isinstance(fallback, BatchedRegionalAttentionContexts):
            raise TypeError("Regional attention fallback has an invalid type.")
        current = self._current.get()
        return fallback if current is None else current

    def require_current(self) -> BatchedRegionalAttentionContexts:
        """Return active contexts or reject execution outside a model call."""

        current = self._current.get()
        if current is None:
            raise RuntimeError(
                "Regional attention contexts are unavailable outside the "
                "SimpleSyrup diffusion wrapper."
            )
        return current

    @contextmanager
    def activate(
        self,
        contexts: BatchedRegionalAttentionContexts,
    ) -> Iterator[None]:
        """Publish one aligned context batch for exactly one diffusion call."""

        if not isinstance(contexts, BatchedRegionalAttentionContexts):
            raise TypeError("Regional attention contexts have an invalid type.")
        token = self._current.set(contexts)
        try:
            yield
        finally:
            self._current.reset(token)
