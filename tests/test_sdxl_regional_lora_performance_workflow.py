# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify image-free matched SDXL regional-LoRA benchmark graphs."""

from tools.sdxl_full_strength_lora_fidelity.cases import (
    CharacterFidelitySpec,
    fidelity_cases,
)
from tools.sdxl_regional_lora_performance.workflow import build_performance_workflow


def test_performance_projection_preserves_sampler_and_removes_image_work() -> None:
    """Keep both execution modes measurable without decoding or saving images."""

    character = CharacterFidelitySpec(
        "character", "Character", "adapter.safetensors", "trigger", "trigger"
    )
    for case in fidelity_cases((character,)):
        built = build_performance_workflow(
            run_id="run",
            checkpoint_name="checkpoint.safetensors",
            mask_name="mask.png",
            base_positive_g="positive g",
            base_positive_l="positive l",
            negative_g="negative g",
            negative_l="negative l",
            case=case,
        )
        classes = tuple(str(node["class_type"]) for node in built.prompt.values())
        assert "SaveImage" not in classes
        assert "VAEDecode" not in classes
        assert "SimpleSyrupBenchmark.ReadMetrics" in classes
        assert ("KSampler" in classes) != (
            "SimpleSyrup.KSamplerAttentionCoupling" in classes
        )
