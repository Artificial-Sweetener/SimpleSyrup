# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify one exact benchmark model-call capture decision."""

from __future__ import annotations

import pytest

from tools.attention_coupling_benchmark.comfy_probe.indexed_call_capture import (
    IndexedModelCallCapture,
)


def test_capture_admits_only_the_declared_one_based_call() -> None:
    """Skip warm calls and select the configured call exactly once."""

    capture = IndexedModelCallCapture(3)

    assert [capture.admit_next() for _ in range(5)] == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert capture.call_count == 5


@pytest.mark.parametrize("call_index", [0, -1])
def test_capture_rejects_nonpositive_indices(call_index: int) -> None:
    """Reject a capture target outside one-based model-call order."""

    with pytest.raises(ValueError, match="positive"):
        IndexedModelCallCapture(call_index)
