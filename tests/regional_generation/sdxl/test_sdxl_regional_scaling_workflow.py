# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify prepared regional scaling graph topology."""

from __future__ import annotations

from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_regional_lora_performance.scaling_cases import regional_scaling_cases
from tools.sdxl_regional_lora_performance.scaling_workflow import (
    build_regional_scaling_steady_state_workflow,
)


def test_scaling_workflows_preserve_sampler_and_exact_hook_cardinality() -> None:
    """Build zero, one, and four uses without instrumentation or image work."""

    cases = regional_scaling_cases(
        _prompts(),
        left_trigger_g="left trigger g",
        left_trigger_l="left trigger l",
        right_trigger_g="right trigger g",
        right_trigger_l="right trigger l",
        style_trigger_g="style trigger g",
        style_trigger_l="style trigger l",
    )
    for declared in cases:
        built = build_regional_scaling_steady_state_workflow(
            checkpoint_name="generic-checkpoint.safetensors",
            mask_names=("left.png", "right.png"),
            declared=declared,
        )
        class_types = tuple(node["class_type"] for node in built.prompt.values())
        assert class_types.count("CreateHookLora") == declared.adapter_count
        assert class_types.count("SimpleSyrup.KSamplerAttentionCoupling") == 1
        assert class_types.count("SimpleSyrupBenchmark.CompleteLatent") == 1
        assert "SimpleSyrupBenchmark.InstrumentModel" not in class_types
        assert "SimpleSyrupBenchmark.ReadMetrics" not in class_types
        assert "VAEDecode" not in class_types
        assert "SaveImage" not in class_types


def _prompts() -> SdxlVisualPromptSet:
    """Return one generic prompt fixture shared by all scaling profiles."""

    return SdxlVisualPromptSet(
        base_positive_g="base g",
        base_positive_l="base l",
        base_negative_g="negative g",
        base_negative_l="negative l",
        left_positive_g="left g",
        left_positive_l="left l",
        right_positive_g="right g",
        right_positive_l="right l",
        left_negative_g="left negative g",
        left_negative_l="left negative l",
        right_negative_g="right negative g",
        right_negative_l="right negative l",
    )
