# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify cold upstream node/stage decomposition."""

from __future__ import annotations

import pytest

from tools.comfy_integration.execution_trace import (
    ComfyExecutionTraceResult,
    ComfyNodeExecutionTiming,
)
from tools.sdxl_cold_upstream_trace.results import attribute_cold_upstream
from tools.sdxl_regional_lora_performance.cold_path_results import (
    ColdPathStageTiming,
    SdxlColdPathTiming,
)


def test_attribution_separates_upstream_sampler_and_terminal_intervals() -> None:
    """Explain cold wall time without double-counting nested sampler stages."""

    trace = ComfyExecutionTraceResult(
        prompt_id="prompt",
        submission_started_at_ns=1_000_000_000,
        completed_at_ns=1_200_000_000,
        cached_node_ids=("cached",),
        nodes=(
            ComfyNodeExecutionTiming("a", "Encode", 1_010_000_000, 40.0),
            ComfyNodeExecutionTiming("s", "Sampler", 1_050_000_000, 120.0),
            ComfyNodeExecutionTiming("t", "Terminal", 1_170_000_000, 30.0),
        ),
        history={},
    )
    cold = SdxlColdPathTiming(
        run_id="cold",
        seed=1,
        runtime_ms=200.0,
        stages=(
            ColdPathStageTiming("admission_resolution", 10.0, {}),
            ColdPathStageTiming("template_preparation", 20.0, {}),
            ColdPathStageTiming("model_residency", 10.0, {}),
            ColdPathStageTiming("sampling", 60.0, {}),
            ColdPathStageTiming("variant_materialization", 5.0, {}),
            ColdPathStageTiming("variant_shell", 1.0, {}),
        ),
        model_call_count=30,
        peak_vram_bytes=1,
    )

    result = attribute_cold_upstream(trace, cold, sampler_node_id="s")

    assert result.submission_to_first_node_ms == 10.0
    assert result.pre_sampler_node_ms == 40.0
    assert result.sampler_node_ms == 120.0
    assert result.sampler_instrumented_ms == 100.0
    assert result.sampler_unattributed_ms == 20.0
    assert result.post_sampler_node_ms == 30.0
    assert result.trace_residual_ms == 0.0
    assert result.cold_unattributed_ms == 100.0
    assert result.named_upstream_fraction == 0.4
    assert result.upstream_class_totals_ms == {"Encode": 40.0}


def test_attribution_requires_one_traced_sampler() -> None:
    """Fail closed when graph identity cannot isolate the sampler interval."""

    trace = ComfyExecutionTraceResult(
        "prompt",
        1,
        2,
        (),
        (ComfyNodeExecutionTiming("other", "Other", 1, 0.0),),
        {},
    )
    cold = SdxlColdPathTiming("cold", 1, 1.0, (), 30, 1)

    with pytest.raises(ValueError, match="exactly one sampler"):
        attribute_cold_upstream(trace, cold, sampler_node_id="sampler")
