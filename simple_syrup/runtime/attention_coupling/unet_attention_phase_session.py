# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Publish the standard-UNet attention phase for one model call."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from .unet_attention_phase import StandardUnetAttentionPhase
from .unet_attention_phase_diagnostics import (
    STANDARD_UNET_ATTENTION_PHASE_DIAGNOSTICS_EMITTER,
    StandardUnetAttentionPhaseDiagnostic,
    StandardUnetAttentionPhaseDiagnosticsEmitter,
)
from .unet_attention_phase_schedule import (
    STANDARD_UNET_ATTENTION_PHASE_SCHEDULE,
    StandardUnetAttentionPhaseSchedule,
)


class StandardUnetAttentionPhaseSession:
    """Own call-local attention phase selection and lifetime."""

    def __init__(
        self,
        schedule: StandardUnetAttentionPhaseSchedule = (
            STANDARD_UNET_ATTENTION_PHASE_SCHEDULE
        ),
        diagnostics: StandardUnetAttentionPhaseDiagnosticsEmitter = (
            STANDARD_UNET_ATTENTION_PHASE_DIAGNOSTICS_EMITTER
        ),
    ) -> None:
        """Retain the authoritative schedule and focused diagnostics owner."""

        if not isinstance(schedule, StandardUnetAttentionPhaseSchedule):
            raise TypeError("Standard UNet attention phase schedule is invalid.")
        if not isinstance(diagnostics, StandardUnetAttentionPhaseDiagnosticsEmitter):
            raise TypeError("Standard UNet attention phase diagnostics are invalid.")
        self._schedule = schedule
        self._diagnostics = diagnostics
        self._current: ContextVar[StandardUnetAttentionPhase | None] = ContextVar(
            "simple_syrup_standard_unet_attention_phase",
            default=None,
        )

    @contextmanager
    def activate(self, transformer_options: dict[str, object]) -> Iterator[None]:
        """Publish the exact phase around one standard-UNet model call."""

        phase = self._schedule.resolve(transformer_options)
        token = self._current.set(phase)
        try:
            self._diagnostics.emit(StandardUnetAttentionPhaseDiagnostic(phase))
            yield
        finally:
            self._current.reset(token)

    def require_current(self) -> StandardUnetAttentionPhase:
        """Return the active phase or reject use outside its model call."""

        phase = self._current.get()
        if phase is None:
            raise RuntimeError("Standard UNet attention phase is inactive.")
        return phase


STANDARD_UNET_ATTENTION_PHASE_SESSION = StandardUnetAttentionPhaseSession()
