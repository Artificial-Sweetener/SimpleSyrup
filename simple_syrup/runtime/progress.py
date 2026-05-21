# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Small progress reporting boundary for ComfyUI node work."""

from __future__ import annotations

from importlib import import_module
from typing import Protocol, cast


class ProgressReporter(Protocol):
    """Progress sink compatible with ComfyUI progress bars."""

    def update(self, value: int) -> None:
        """Advance progress by value."""


class NullProgressReporter:
    """Ignore progress updates when no UI progress sink is available."""

    def update(self, value: int) -> None:
        """Ignore progress updates."""


def create_comfy_progress(total: int) -> ProgressReporter:
    """Return a ComfyUI progress bar for total units of work."""

    if total <= 0:
        return NullProgressReporter()
    comfy_utils = import_module("comfy.utils")
    return cast(ProgressReporter, comfy_utils.ProgressBar(total))
