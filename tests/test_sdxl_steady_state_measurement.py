# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify synchronized cache-realistic SDXL timing decoding."""

from __future__ import annotations

import pytest

from tools.sdxl_regional_lora_performance.steady_state_measurement import (
    SdxlSteadyStateTiming,
    decode_steady_state_timing,
    median_steady_state_runtime_ms,
)


def test_decode_steady_state_timing_uses_synchronized_completion() -> None:
    """Calculate elapsed wall time from one exact terminal timestamp."""

    timing = decode_steady_state_timing(
        {"outputs": {"9": {"benchmark_completion": [{"completed_at_ns": 3_500_000}]}}},
        completion_node_id="9",
        started_at_ns=1_000_000,
        seed=42,
    )

    assert timing == SdxlSteadyStateTiming(runtime_ms=2.5, seed=42)


def test_decode_steady_state_timing_rejects_missing_completion() -> None:
    """Reject histories that cannot prove synchronized completion."""

    with pytest.raises(ValueError, match="completion node"):
        decode_steady_state_timing(
            {"outputs": {}},
            completion_node_id="9",
            started_at_ns=1,
            seed=42,
        )


def test_median_steady_state_runtime_requires_three_repeats() -> None:
    """Summarize only a stable three-or-more observation block."""

    values = tuple(
        SdxlSteadyStateTiming(runtime_ms=value, seed=index)
        for index, value in enumerate((9.0, 7.0, 8.0), start=1)
    )

    assert median_steady_state_runtime_ms(values) == 8.0
    with pytest.raises(ValueError, match="requires three repeats"):
        median_steady_state_runtime_ms(values[:2])
