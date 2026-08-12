# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Publish one task-local regional LoRA schedule resolution per model call."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from ..regional_lora_schedule_resolution import RegionalLoraScheduleResolution


class AnimaRegionalLoraScheduleContext:
    """Own exact active adapter strengths during nested Anima block execution."""

    def __init__(self) -> None:
        """Create an empty task-local schedule slot."""

        self._current: ContextVar[RegionalLoraScheduleResolution | None] = ContextVar(
            "simple_syrup_anima_regional_lora_schedule",
            default=None,
        )

    def require_current(self) -> RegionalLoraScheduleResolution:
        """Return the active resolution or fail outside its model-call scope."""

        current = self._current.get()
        if current is None:
            raise RuntimeError(
                "Anima regional LoRA schedule is unavailable outside the "
                "schedule diffusion wrapper."
            )
        return current

    @contextmanager
    def activate(
        self,
        resolution: RegionalLoraScheduleResolution,
    ) -> Iterator[None]:
        """Publish one resolution for exactly one nested model call."""

        if not isinstance(resolution, RegionalLoraScheduleResolution):
            raise TypeError("Anima regional LoRA schedule resolution is invalid.")
        token = self._current.set(resolution)
        try:
            yield
        finally:
            self._current.reset(token)


ANIMA_REGIONAL_LORA_SCHEDULE_CONTEXT = AnimaRegionalLoraScheduleContext()
