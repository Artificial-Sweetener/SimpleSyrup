# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define regional-character and global-style comparison cases."""

from __future__ import annotations

from dataclasses import dataclass

PRIMARY_ADAPTER_NAME = r"Anima\style\adapter-a.safetensors"


@dataclass(frozen=True, slots=True)
class GlobalStyleCharacterCase:
    """Describe one whole-image style strength with a regional character."""

    case_id: str
    label: str
    global_style_strength: float | None


def cases() -> tuple[GlobalStyleCharacterCase, ...]:
    """Return the character control and two whole-image style strengths."""

    return (
        GlobalStyleCharacterCase(
            "character-right-only",
            "CHARACTER RIGHT 1.0 — no global style",
            None,
        ),
        GlobalStyleCharacterCase(
            "global-style-050-character-right",
            "GLOBAL STYLE 0.5 + CHARACTER RIGHT 1.0",
            0.5,
        ),
        GlobalStyleCharacterCase(
            "global-style-100-character-right",
            "GLOBAL STYLE 1.0 + CHARACTER RIGHT 1.0",
            1.0,
        ),
    )
