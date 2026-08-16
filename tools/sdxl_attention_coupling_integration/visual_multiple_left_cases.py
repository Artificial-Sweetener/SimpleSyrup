# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own multiple-adapter left-region SDXL visual cases and diagnostics."""

from __future__ import annotations

from .visual_adapter_selections import STYLE_SELECTION
from .visual_case_model import RegionalVisualAdapter, SdxlVisualCase
from .visual_inventory import SdxlVisualInventory

_LAYOUT_FIRST_SCHEDULE = ((0.0, 0.0), (0.3, 0.35), (0.45, 1.0))
_MIDPOINT_LAYOUT_SCHEDULE = ((0.0, 0.0), (0.15, 0.35), (0.30, 1.0))
_MINIMAL_LAYOUT_SCHEDULE = ((0.0, 0.0), (0.05, 0.35), (0.15, 1.0))


def multiple_left_cases(
    inventory: SdxlVisualInventory,
    *,
    left_character: RegionalVisualAdapter,
) -> tuple[SdxlVisualCase, ...]:
    """Return the causal controls and ordered multiple-adapter acceptance case."""

    left_l = f"{inventory.left_character.prompt_l}, {inventory.style.prompt_l}"
    left_g = f"{inventory.left_character.prompt_g}, {inventory.style.prompt_g}"
    style = RegionalVisualAdapter(STYLE_SELECTION, 0.55, 0.55)
    return (
        _case(
            inventory,
            "multiple-left-style-prompt-control",
            "Character-and-style prompt on left without adapters",
            left_l=left_l,
            left_g=left_g,
        ),
        _case(
            inventory,
            "multiple-left-style-only-control",
            "Style adapter on left with character-and-style prompt",
            left_l=left_l,
            left_g=left_g,
            left_adapters=(style,),
        ),
        _case(
            inventory,
            "multiple-left-model-only-style",
            "Model-only style adapter on left with identical prompt",
            left_l=left_l,
            left_g=left_g,
            left_adapters=(RegionalVisualAdapter(STYLE_SELECTION, 0.55, 0.0),),
            regional_prompt_weight=1.0,
        ),
        _case(
            inventory,
            "multiple-left-scheduled-style-only",
            "Layout-first style adapter on left with character-and-style prompt",
            left_l=left_l,
            left_g=left_g,
            left_adapters=(
                RegionalVisualAdapter(
                    STYLE_SELECTION,
                    0.55,
                    0.55,
                    _LAYOUT_FIRST_SCHEDULE,
                ),
            ),
            regional_prompt_weight=1.0,
        ),
        _case(
            inventory,
            "multiple-left-midpoint-style-only",
            "Earlier layout-first style adapter on left",
            left_l=left_l,
            left_g=left_g,
            left_adapters=(
                RegionalVisualAdapter(
                    STYLE_SELECTION,
                    0.55,
                    0.55,
                    _MIDPOINT_LAYOUT_SCHEDULE,
                ),
            ),
            regional_prompt_weight=1.0,
        ),
        _case(
            inventory,
            "multiple-left-minimal-window-style-only",
            "Minimal-window style adapter on left",
            left_l=left_l,
            left_g=left_g,
            left_adapters=(
                RegionalVisualAdapter(
                    STYLE_SELECTION,
                    0.55,
                    0.55,
                    _MINIMAL_LAYOUT_SCHEDULE,
                ),
            ),
            regional_prompt_weight=1.0,
        ),
        _case(
            inventory,
            "multiple-left",
            "Character and style adapters on left",
            left_l=left_l,
            left_g=left_g,
            left_adapters=(left_character, style),
        ),
    )


def _case(
    inventory: SdxlVisualInventory,
    case_id: str,
    label: str,
    *,
    left_l: str,
    left_g: str,
    left_adapters: tuple[RegionalVisualAdapter, ...] = (),
    regional_prompt_weight: float = 0.4,
) -> SdxlVisualCase:
    """Build one multiple-left case with exact shared prompt geometry."""

    return SdxlVisualCase(
        case_id,
        label,
        left_l=left_l,
        left_g=left_g,
        left_adapters=left_adapters,
        regional_prompt_weight=regional_prompt_weight,
    )
