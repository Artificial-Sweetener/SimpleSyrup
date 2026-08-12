# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own task-local Anima cross-attention branch invocation identity."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from .anima_branch_batch import AnimaRegionalBranchInvocation


class AnimaCrossAttentionInvocationContext:
    """Own task-local branch identity during the duplicated attention call."""

    def __init__(self) -> None:
        """Create an empty task-local invocation slot."""

        self._current: ContextVar[AnimaRegionalBranchInvocation | None] = ContextVar(
            "simple_syrup_anima_cross_attention_invocation",
            default=None,
        )

    def current_or_none(self) -> AnimaRegionalBranchInvocation | None:
        """Return the active branch batch when called from cross-attention."""

        return self._current.get()

    def require_current(self) -> AnimaRegionalBranchInvocation:
        """Return the active branch batch or fail outside its owned scope."""

        invocation = self.current_or_none()
        if invocation is None:
            raise RuntimeError(
                "Anima cross-attention invocation is unavailable outside the "
                "regional attention patch."
            )
        return invocation

    @contextmanager
    def activate(self, invocation: AnimaRegionalBranchInvocation) -> Iterator[None]:
        """Publish one invocation for exactly one original attention call."""

        token = self._current.set(invocation)
        try:
            yield
        finally:
            self._current.reset(token)


ANIMA_CROSS_ATTENTION_INVOCATION_CONTEXT = AnimaCrossAttentionInvocationContext()
