# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decompose cold wall time across traced Comfy nodes and sampler stages."""

from __future__ import annotations

from dataclasses import dataclass

from tools.comfy_integration.execution_trace import ComfyExecutionTraceResult
from tools.sdxl_regional_lora_performance.cold_path_results import (
    SdxlColdPathTiming,
)

_TOP_LEVEL_STAGES = frozenset(
    {
        "admission_resolution",
        "template_preparation",
        "model_residency",
        "sampling",
    }
)


@dataclass(frozen=True, slots=True)
class SdxlColdUpstreamAttribution:
    """Retain one complete node/stage decomposition of cold wall time."""

    submission_to_first_node_ms: float
    pre_sampler_node_ms: float
    sampler_node_ms: float
    sampler_instrumented_ms: float
    sampler_unattributed_ms: float
    post_sampler_node_ms: float
    trace_residual_ms: float
    cold_unattributed_ms: float
    named_upstream_fraction: float
    upstream_class_totals_ms: dict[str, float]
    all_class_totals_ms: dict[str, float]


def attribute_cold_upstream(
    trace: ComfyExecutionTraceResult,
    cold: SdxlColdPathTiming,
    *,
    sampler_node_id: str,
) -> SdxlColdUpstreamAttribution:
    """Decompose traced graph work around one instrumented sampler node."""

    sampler_indices = tuple(
        index
        for index, timing in enumerate(trace.nodes)
        if timing.node_id == sampler_node_id
    )
    if len(sampler_indices) != 1:
        raise ValueError("Cold upstream trace requires exactly one sampler node.")
    sampler_index = sampler_indices[0]
    pre_sampler = trace.nodes[:sampler_index]
    sampler = trace.nodes[sampler_index]
    post_sampler = trace.nodes[sampler_index + 1 :]
    first_started_at_ns = trace.nodes[0].started_at_ns
    submission_to_first_ms = (
        first_started_at_ns - trace.submission_started_at_ns
    ) / 1_000_000.0
    stage_totals: dict[str, float] = {}
    for stage in cold.stages:
        stage_totals[stage.stage] = (
            stage_totals.get(stage.stage, 0.0) + stage.elapsed_ms
        )
    sampler_instrumented_ms = sum(
        stage_totals.get(stage, 0.0) for stage in _TOP_LEVEL_STAGES
    )
    cold_unattributed_ms = cold.runtime_ms - sampler_instrumented_ms
    pre_sampler_ms = sum(node.elapsed_ms for node in pre_sampler)
    post_sampler_ms = sum(node.elapsed_ms for node in post_sampler)
    sampler_unattributed_ms = sampler.elapsed_ms - sampler_instrumented_ms
    observed_components = (
        submission_to_first_ms + pre_sampler_ms + sampler.elapsed_ms + post_sampler_ms
    )
    upstream_totals: dict[str, float] = {}
    for node in pre_sampler:
        upstream_totals[node.class_type] = (
            upstream_totals.get(node.class_type, 0.0) + node.elapsed_ms
        )
    return SdxlColdUpstreamAttribution(
        submission_to_first_node_ms=submission_to_first_ms,
        pre_sampler_node_ms=pre_sampler_ms,
        sampler_node_ms=sampler.elapsed_ms,
        sampler_instrumented_ms=sampler_instrumented_ms,
        sampler_unattributed_ms=sampler_unattributed_ms,
        post_sampler_node_ms=post_sampler_ms,
        trace_residual_ms=trace.elapsed_ms - observed_components,
        cold_unattributed_ms=cold_unattributed_ms,
        named_upstream_fraction=(
            pre_sampler_ms / cold_unattributed_ms if cold_unattributed_ms > 0.0 else 0.0
        ),
        upstream_class_totals_ms=upstream_totals,
        all_class_totals_ms=trace.class_totals_ms(),
    )
