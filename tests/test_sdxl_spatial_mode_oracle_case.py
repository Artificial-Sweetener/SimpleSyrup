# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lock the full-strength SDXL spatial-mode oracle declaration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from sdxl_visual_test_inventory import visual_inventory

from tools.sdxl_attention_coupling_integration.visual_case_model import VisualMode
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_spatial_mode_oracle import (
    SDXL_SPATIAL_MODE_ORACLE_ID,
    spatial_mode_oracle_cases,
)
from tools.sdxl_two_character_oracle import two_character_oracle_cases


def test_spatial_oracle_changes_only_identity_label_and_declared_modes(
    tmp_path: Path,
) -> None:
    """Keep every accepted source input exact while enabling refinements."""

    inventory = visual_inventory(tmp_path)
    prompts = SdxlVisualPromptSet(*(f"authored-prompt-{index}" for index in range(12)))
    (accepted_source,) = two_character_oracle_cases(inventory, prompts)
    (spatial_case,) = spatial_mode_oracle_cases(inventory, prompts)

    assert spatial_case == replace(
        accepted_source,
        case_id=SDXL_SPATIAL_MODE_ORACLE_ID,
        label="SPATIAL ORACLE — full 1024; tiled and Contextual 1.5x",
        modes=(VisualMode.FULL, VisualMode.TILED, VisualMode.CONTEXTUAL),
    )
