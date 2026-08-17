# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load portable fonts for generated benchmark labels."""

from __future__ import annotations

from PIL import ImageFont


def load_label_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return Pillow's bundled font at the requested readable label size."""

    if size <= 0:
        raise ValueError("Label font size must be positive.")
    return ImageFont.load_default(size=size)
