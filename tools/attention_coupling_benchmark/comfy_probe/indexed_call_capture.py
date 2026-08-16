# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Select exactly one indexed model call for benchmark capture."""

from __future__ import annotations


class IndexedModelCallCapture:
    """Own one-based call counting and single-capture admission."""

    def __init__(self, call_index: int) -> None:
        """Require and retain one positive one-based capture index."""

        if isinstance(call_index, bool) or not isinstance(call_index, int):
            raise TypeError("Profiled model call index must be an integer.")
        if call_index < 1:
            raise ValueError("Profiled model call index must be positive.")
        self._call_index = call_index
        self._call_count = 0
        self._captured = False

    def admit_next(self) -> bool:
        """Return true exactly once when the configured call is reached."""

        self._call_count += 1
        if self._captured or self._call_count != self._call_index:
            return False
        self._captured = True
        return True

    @property
    def call_count(self) -> int:
        """Return the number of calls observed so far."""

        return self._call_count
