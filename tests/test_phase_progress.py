# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for standard Comfy phase-progress publication."""

from __future__ import annotations

from types import ModuleType

import pytest

from simple_syrup.runtime.progress import ComfyPhaseProgressReporter


def test_comfy_phase_progress_publishes_boundaries_and_terminal_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Named phases should use standard Comfy progress and finish terminally."""

    updates: list[tuple[int, int | None]] = []

    class _ProgressBar:
        """Record standard Comfy absolute progress updates."""

        def __init__(self, total: int) -> None:
            """Record the configured total."""

            assert total == 4

        def update_absolute(self, value: int, total: int | None = None) -> None:
            """Record one absolute progress update."""

            updates.append((value, total))

    comfy_utils = ModuleType("comfy.utils")
    comfy_utils.ProgressBar = _ProgressBar  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "simple_syrup.runtime.progress.import_module",
        lambda _name: comfy_utils,
    )
    reporter = ComfyPhaseProgressReporter(
        operation="model_load",
        subject="model-a",
        total_phases=4,
    )

    reporter.advance("resolving")
    reporter.advance("cache_hit")
    reporter.advance("completed")

    assert updates == [(1, 4), (2, 4), (4, 4)]


def test_comfy_phase_progress_does_not_present_failure_as_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed phase should leave room for Comfy's terminal error state."""

    updates: list[tuple[int, int | None]] = []

    class _ProgressBar:
        """Record standard Comfy absolute progress updates."""

        def __init__(self, _total: int) -> None:
            """Accept the configured total."""

        def update_absolute(self, value: int, total: int | None = None) -> None:
            """Record one absolute progress update."""

            updates.append((value, total))

    comfy_utils = ModuleType("comfy.utils")
    comfy_utils.ProgressBar = _ProgressBar  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "simple_syrup.runtime.progress.import_module",
        lambda _name: comfy_utils,
    )
    reporter = ComfyPhaseProgressReporter(
        operation="model_load",
        subject="model-a",
        total_phases=4,
    )

    reporter.advance("resolving")
    reporter.advance("failed")

    assert updates == [(1, 4), (2, 4)]
