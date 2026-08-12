# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Publish one task-local Anima composition phase per model call."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from .anima_composition_phase import AnimaCompositionPhase


class AnimaCompositionPhaseContext:
    """Own coordinated phase state for nested Anima collaborators."""

    def __init__(self) -> None:
        """Create an empty task-local phase slot."""

        self._current: ContextVar[AnimaCompositionPhase | None] = ContextVar(
            "simple_syrup_anima_composition_phase",
            default=None,
        )

    def require_current(self) -> AnimaCompositionPhase:
        """Return the phase or reject execution outside its wrapper."""

        phase = self._current.get()
        if phase is None:
            raise RuntimeError(
                "Anima composition phase is unavailable outside its diffusion wrapper."
            )
        return phase

    @contextmanager
    def activate(self, phase: AnimaCompositionPhase) -> Iterator[None]:
        """Publish one validated phase for a nested model call."""

        if not isinstance(phase, AnimaCompositionPhase):
            raise TypeError("Anima composition phase is invalid.")
        token = self._current.set(phase)
        try:
            yield
        finally:
            self._current.reset(token)


ANIMA_COMPOSITION_PHASE_CONTEXT = AnimaCompositionPhaseContext()
