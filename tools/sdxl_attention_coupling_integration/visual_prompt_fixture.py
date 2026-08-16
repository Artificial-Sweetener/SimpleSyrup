# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt one external parity prompt fixture to SDXL visual cases."""

from __future__ import annotations

from pathlib import Path

from tools.sdxl_attention_couple_parity.cases import load_parity_case

from .visual_lora_baseline_cases import SdxlVisualPromptSet


def load_visual_prompt_set(path: Path) -> SdxlVisualPromptSet:
    """Load and preserve every authored global and regional G/L prompt."""

    prompts = load_parity_case(path)
    return SdxlVisualPromptSet(
        base_positive_g=prompts.base_positive_g,
        base_positive_l=prompts.base_positive_l,
        base_negative_g=prompts.base_negative_g,
        base_negative_l=prompts.base_negative_l,
        left_positive_g=prompts.left_positive_g,
        left_positive_l=prompts.left_positive_l,
        right_positive_g=prompts.right_positive_g,
        right_positive_l=prompts.right_positive_l,
        left_negative_g=prompts.left_negative_g,
        left_negative_l=prompts.left_negative_l,
        right_negative_g=prompts.right_negative_g,
        right_negative_l=prompts.right_negative_l,
    )
