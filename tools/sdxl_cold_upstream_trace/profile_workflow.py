# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Substitute only the dev-only profiled sampler in one cold trace graph."""

from __future__ import annotations

from tools.sdxl_attention_coupling_integration.visual_case_model import SdxlVisualCase
from tools.sdxl_regional_lora_performance.cold_path_workflow import (
    BuiltSdxlColdPathWorkflow,
    build_sdxl_cold_path_workflow,
)

_PRODUCTION_SAMPLER = "SimpleSyrup.KSamplerAttentionCoupling"
_PROFILED_SAMPLER = "SimpleSyrupBenchmark.ProfiledKSamplerAttentionCoupling"


def build_profiled_cold_path_workflow(
    *,
    checkpoint_name: str,
    mask_names: tuple[str, str],
    case: SdxlVisualCase,
) -> BuiltSdxlColdPathWorkflow:
    """Return the unchanged cold graph with one sampler identity substitution."""

    built = build_sdxl_cold_path_workflow(
        checkpoint_name=checkpoint_name,
        mask_names=mask_names,
        case=case,
    )
    sampler = built.prompt.get(built.sampler_node_id)
    if (
        not isinstance(sampler, dict)
        or sampler.get("class_type") != _PRODUCTION_SAMPLER
    ):
        raise ValueError("Cold trace lost its production Attention Coupling sampler.")
    sampler["class_type"] = _PROFILED_SAMPLER
    return built
