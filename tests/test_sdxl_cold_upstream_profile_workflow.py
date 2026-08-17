# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the cold profile graph substitutes only its sampler identity."""

from __future__ import annotations

from tools.sdxl_attention_coupling_integration.visual_case_model import (
    RegionalVisualAdapter,
    SdxlVisualCase,
)
from tools.sdxl_cold_upstream_trace.profile_workflow import (
    build_profiled_cold_path_workflow,
)
from tools.sdxl_regional_lora_performance.cold_path_workflow import (
    build_sdxl_cold_path_workflow,
)


def test_profile_workflow_changes_only_the_sampler_class() -> None:
    """Preserve every graph input, edge, node id, and terminal unchanged."""

    case = _case()
    production = build_sdxl_cold_path_workflow(
        checkpoint_name="checkpoint",
        mask_names=("left-mask", "right-mask"),
        case=case,
    )
    profiled = build_profiled_cold_path_workflow(
        checkpoint_name="checkpoint",
        mask_names=("left-mask", "right-mask"),
        case=case,
    )

    assert production.sampler_node_id == profiled.sampler_node_id
    assert production.capture_node_id == profiled.capture_node_id
    assert production.terminal_node_id == profiled.terminal_node_id
    profiled.prompt[profiled.sampler_node_id]["class_type"] = (
        "SimpleSyrup.KSamplerAttentionCoupling"
    )
    assert profiled.prompt == production.prompt


def _case() -> SdxlVisualCase:
    """Return one model-identity-neutral two-adapter case."""

    return SdxlVisualCase(
        "profile",
        "Profile",
        base_positive_g="two subjects",
        base_positive_l="two subjects",
        left_g="left traits",
        left_l="left traits",
        right_g="right traits",
        right_l="right traits",
        left_adapters=(RegionalVisualAdapter("adapter-a", 1.0, 1.0),),
        right_adapters=(RegionalVisualAdapter("adapter-b", 1.0, 1.0),),
        regional_prompt_weight=1.0,
    )
