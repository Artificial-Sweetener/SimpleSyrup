# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own immutable adapter strengths and schedules for SDXL visual proofs."""

from __future__ import annotations

from .visual_adapter_selections import (
    LEFT_CHARACTER_SELECTION,
    RIGHT_CHARACTER_SELECTION,
    STYLE_SELECTION,
)
from .visual_case_model import GlobalVisualAdapter, RegionalVisualAdapter

CHARACTER_LAYOUT_FIRST_SCHEDULE = (
    (0.0, 0.0),
    (0.3, 0.35),
    (0.45, 1.0),
)


def left_character_adapter() -> RegionalVisualAdapter:
    """Return the canonical left-character visual adapter declaration."""

    return RegionalVisualAdapter(
        LEFT_CHARACTER_SELECTION,
        0.45,
        0.45,
        CHARACTER_LAYOUT_FIRST_SCHEDULE,
    )


def right_character_adapter() -> RegionalVisualAdapter:
    """Return the canonical right-character visual adapter declaration."""

    return RegionalVisualAdapter(
        RIGHT_CHARACTER_SELECTION,
        0.45,
        0.45,
        CHARACTER_LAYOUT_FIRST_SCHEDULE,
    )


def global_style_adapter() -> GlobalVisualAdapter:
    """Return the canonical ordinary whole-image style declaration."""

    return GlobalVisualAdapter(STYLE_SELECTION, 0.65)
