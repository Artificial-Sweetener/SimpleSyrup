# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own the full-strength SDXL full, tiled, and Contextual oracle case."""

from __future__ import annotations

from dataclasses import replace

from tools.sdxl_attention_coupling_integration.visual_case_model import (
    SdxlVisualCase,
    VisualMode,
)
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_two_character_oracle import two_character_oracle_cases

SDXL_SPATIAL_MODE_ORACLE_ID = "sdxl-spatial-mode-oracle"


def spatial_mode_oracle_cases(
    inventory: SdxlVisualInventory,
    prompts: SdxlVisualPromptSet,
) -> tuple[SdxlVisualCase, ...]:
    """Extend the accepted full source with both 1.5x refinement modes."""

    (accepted_source,) = two_character_oracle_cases(inventory, prompts)
    return (
        replace(
            accepted_source,
            case_id=SDXL_SPATIAL_MODE_ORACLE_ID,
            label="SPATIAL ORACLE — full 1024; tiled and Contextual 1.5x",
            modes=(VisualMode.FULL, VisualMode.TILED, VisualMode.CONTEXTUAL),
        ),
    )
