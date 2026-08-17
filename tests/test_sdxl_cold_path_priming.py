# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify historical-context SDXL cold-path primers."""

from __future__ import annotations

from tools.sdxl_attention_coupling_integration.visual_case_model import (
    RegionalVisualAdapter,
    SdxlVisualCase,
)
from tools.sdxl_regional_lora_performance.cold_path_priming import (
    build_cold_path_primers,
)


def test_primers_preserve_global_left_right_order_without_images() -> None:
    """Warm only the dependencies present before the historical regional run."""

    primers = build_cold_path_primers(
        checkpoint_name="checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=_case(),
    )

    assert tuple(primer.label for primer in primers) == (
        "global-two-lora-reference",
        "conventional-left-variant",
        "conventional-right-variant",
    )
    for primer in primers:
        class_types = tuple(
            node["class_type"] for node in primer.workflow.prompt.values()
        )
        assert "SimpleSyrupBenchmark.CompleteLatent" in class_types
        assert "VAEDecode" not in class_types
        assert "SaveImage" not in class_types
        assert "SimpleSyrupBenchmark.CaptureColdPathDiagnostics" not in class_types


def _case() -> SdxlVisualCase:
    """Return two anonymous full-strength character adapters."""

    return SdxlVisualCase(
        "cold-primer",
        "Cold primer",
        base_positive_g="two subjects",
        base_positive_l="two subjects",
        left_g="left traits",
        left_l="left traits",
        right_g="right traits",
        right_l="right traits",
        left_adapters=(RegionalVisualAdapter("left.safetensors", 1.0, 1.0),),
        right_adapters=(RegionalVisualAdapter("right.safetensors", 1.0, 1.0),),
        regional_prompt_weight=1.0,
    )
