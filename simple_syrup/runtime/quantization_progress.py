# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Publish quiet byte-weighted quantization progress through ComfyUI nodes."""

from __future__ import annotations

import importlib
from typing import Protocol, cast


class QuantizationProgressReporter(Protocol):
    """Report absolute byte-weighted quantization progress."""

    def start(self, label: str, total: int) -> None:
        """Begin one quantization operation."""

    def advance(self, current: int, total: int) -> None:
        """Report absolute completed work."""

    def finish(self) -> None:
        """Mark the quantization operation complete."""


class _ComfyProgressBar(Protocol):
    """Describe the ComfyUI progress method used by this adapter."""

    def update_absolute(self, value: int, total: int | None = None) -> None:
        """Publish absolute progress."""


class NullQuantizationProgressReporter:
    """Ignore quantization progress outside a ComfyUI execution context."""

    def start(self, label: str, total: int) -> None:
        """Ignore the operation start."""

    def advance(self, current: int, total: int) -> None:
        """Ignore one progress update."""

    def finish(self) -> None:
        """Ignore operation completion."""


class ComfyQuantizationProgressReporter:
    """Attach quantization progress to the currently executing ComfyUI node."""

    def __init__(self) -> None:
        """Create an adapter that allocates its bar only when work starts."""

        self._progress_bar: object | None = None
        self._total = 1

    def start(self, label: str, total: int) -> None:
        """Create a standard ComfyUI progress bar without console chatter."""

        del label
        self._total = max(total, 1)
        comfy_utils = importlib.import_module("comfy.utils")
        self._progress_bar = comfy_utils.ProgressBar(self._total)
        self.advance(0, self._total)

    def advance(self, current: int, total: int) -> None:
        """Publish bounded absolute progress."""

        if self._progress_bar is None:
            return
        self._total = max(total, 1)
        progress_bar = cast(_ComfyProgressBar, self._progress_bar)
        progress_bar.update_absolute(min(max(current, 0), self._total), self._total)

    def finish(self) -> None:
        """Publish completion when an operation was started."""

        if self._progress_bar is None:
            return
        progress_bar = cast(_ComfyProgressBar, self._progress_bar)
        progress_bar.update_absolute(self._total, self._total)
