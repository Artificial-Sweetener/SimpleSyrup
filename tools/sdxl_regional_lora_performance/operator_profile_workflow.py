# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Insert benchmark-only indexed-call profiling into the regional workflow."""

from __future__ import annotations

from pathlib import Path

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.visual_case_model import SdxlVisualCase

from .indexed_profile_workflow import (
    BuiltIndexedOperatorProfileWorkflow,
    build_indexed_operator_profile_workflow,
)
from .two_adapter_workflow import (
    TwoAdapterPerformanceMode,
    build_two_adapter_performance_workflow,
)


def build_sdxl_operator_profile_workflow(
    *,
    run_id: str,
    checkpoint_name: str,
    mask_names: tuple[str, str],
    case: SdxlVisualCase,
    trace_path: Path,
    mode: TwoAdapterPerformanceMode,
    call_index: int,
) -> BuiltIndexedOperatorProfileWorkflow:
    """Profile one indexed call of an otherwise unchanged regional trajectory."""

    built = build_two_adapter_performance_workflow(
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        mask_names=mask_names,
        case=case,
        mode=mode,
    )
    sampler_class = (
        "SimpleSyrup.KSamplerAttentionCoupling"
        if mode is TwoAdapterPerformanceMode.REGIONAL
        else "KSampler"
    )
    sampler_id = _one_node_id(built.prompt, sampler_class)
    return build_indexed_operator_profile_workflow(
        prompt=built.prompt,
        sampler_node_id=sampler_id,
        run_id=f"{run_id}:operator-profile",
        trace_path=trace_path,
        call_index=call_index,
    )


def _one_node_id(prompt: dict[str, JsonObject], class_type: str) -> str:
    """Return exactly one node id for a required class type."""

    matches = tuple(
        node_id
        for node_id, node in prompt.items()
        if node.get("class_type") == class_type
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one {class_type!r} node, found {len(matches)}.")
    return matches[0]
