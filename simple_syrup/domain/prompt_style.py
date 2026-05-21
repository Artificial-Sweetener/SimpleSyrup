# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Format Prompt Control STYLE tags from validated option labels."""

from __future__ import annotations

ENCODE_STYLE_VALUES: dict[str, str] = {
    "A1111": "A1111",
    "Comfy": "comfy",
    "Comfy++": "comfy++",
    "Compel": "compel",
    "Down Weight": "down_weight",
    "Perp": "perp",
}

NORMALIZATION_VALUES: dict[str, str] = {
    "none": "none",
    "length": "length",
    "mean": "mean",
    "length+mean": "length+mean",
}


def format_style_tag(encode_style: str) -> str:
    """Return a Prompt Control STYLE tag for an encode-style option."""

    return f"STYLE({ENCODE_STYLE_VALUES[encode_style]}) "


def format_style_tag_with_normalization(encode_style: str, normalization: str) -> str:
    """Return a Prompt Control STYLE tag with optional normalization."""

    normalization_value = NORMALIZATION_VALUES[normalization]

    if normalization_value == "none":
        return format_style_tag(encode_style)

    return f"STYLE({ENCODE_STYLE_VALUES[encode_style]}, {normalization_value}) "
