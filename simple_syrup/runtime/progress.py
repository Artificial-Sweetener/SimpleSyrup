# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Small progress reporting boundary for ComfyUI node work."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

from ..shared.logging import get_logger

LOGGER = get_logger(__name__)


class ProgressReporter(Protocol):
    """Progress sink compatible with ComfyUI progress bars."""

    def update(self, value: int) -> None:
        """Advance progress by value."""


class NullProgressReporter:
    """Ignore progress updates when no UI progress sink is available."""

    def update(self, value: int) -> None:
        """Ignore progress updates."""


class PhaseProgressReporter(Protocol):
    """Report named phases for a bounded long-running operation."""

    def advance(self, phase: str) -> None:
        """Report that the operation entered ``phase``."""


class NullPhaseProgressReporter:
    """Ignore named phase updates outside a Comfy execution context."""

    def advance(self, phase: str) -> None:
        """Ignore one phase update."""


@dataclass
class ComfyPhaseProgressReporter:
    """Publish truthful phase boundaries through standard Comfy progress events."""

    operation: str
    subject: str
    total_phases: int

    def __post_init__(self) -> None:
        """Create a lazily imported Comfy progress bar and initialize state."""

        if self.total_phases <= 0:
            raise ValueError("total_phases must be positive.")
        comfy_utils = import_module("comfy.utils")
        self._progress_bar = comfy_utils.ProgressBar(self.total_phases)
        self._completed_phases = 0

    def advance(self, phase: str) -> None:
        """Publish one phase transition without inventing intra-phase percentages."""

        self._completed_phases = (
            self.total_phases
            if phase == "completed"
            else min(self.total_phases - 1, self._completed_phases + 1)
        )
        self._progress_bar.update_absolute(
            self._completed_phases,
            self.total_phases,
        )
        LOGGER.info(
            "Long operation entered phase",
            extra={
                "operation": self.operation,
                "subject": self.subject,
                "phase": phase,
                "completed_phases": self._completed_phases,
                "total_phases": self.total_phases,
            },
        )


def create_comfy_progress(total: int) -> ProgressReporter:
    """Return a ComfyUI progress bar for total units of work."""

    if total <= 0:
        return NullProgressReporter()
    comfy_utils = import_module("comfy.utils")
    return cast(ProgressReporter, comfy_utils.ProgressBar(total))


def create_comfy_phase_progress(
    *,
    operation: str,
    subject: str,
    total_phases: int,
) -> PhaseProgressReporter:
    """Return a standard Comfy progress publisher for named phases."""

    return ComfyPhaseProgressReporter(
        operation=operation,
        subject=subject,
        total_phases=total_phases,
    )
