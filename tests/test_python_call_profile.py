# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify benchmark-only Python call profiling."""

from __future__ import annotations

import pytest

from tools.attention_coupling_benchmark.comfy_probe.python_call_profile import (
    PythonCallProfiler,
)


def test_profiler_returns_result_and_stable_function_rows() -> None:
    """Preserve the call result while publishing bounded typed evidence."""

    def profiled() -> int:
        """Return a recognizable result through one nested operation."""

        return sum((1, 2, 3))

    capture = PythonCallProfiler().capture(profiled)

    assert capture.result == 6
    assert 1 <= len(capture.functions) <= 250
    assert any(row["function"] == "profiled" for row in capture.functions)
    for row in capture.functions:
        assert set(row) == {
            "file",
            "line",
            "function",
            "primitive_calls",
            "total_calls",
            "self_time_seconds",
            "cumulative_time_seconds",
        }


def test_profiler_rejects_noncallable_values() -> None:
    """Fail before profiler state exists for an invalid boundary value."""

    with pytest.raises(TypeError, match="requires a callable"):
        PythonCallProfiler().capture(3)  # type: ignore[arg-type]
