# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compose labeled visual evidence for attention-cohesion acceptance runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def compose_sheet(
    *,
    title: str,
    concept: str,
    image_path: Path,
    raw_path: Path,
    previous_path: Path,
    isolated_path: Path,
    solid_path: Path,
    destination: Path,
) -> None:
    """Write a five-panel proof sheet with coverage labels."""

    panels = (
        ("GENERATED IMAGE", image_path, False),
        ("RAW ATTENTION ALPHA", raw_path, True),
        ("PREVIOUS AGGREGATE", previous_path, True),
        ("CONCEPT ISOLATION", isolated_path, True),
        ("ISOLATION + SOLIDITY", solid_path, True),
    )
    panel_width = 320
    panel_height = 426
    header_height = 112
    sheet = Image.new(
        "RGB", (panel_width * len(panels), header_height + panel_height), "#101216"
    )
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(30)
    label_font = load_font(18)
    detail_font = load_font(15)
    draw.text((18, 12), title, fill="white", font=title_font)
    draw.text(
        (18, 54),
        f'Concept: "{concept}"  |  Same generation and shared fast capture',
        fill="#b9c1cc",
        font=detail_font,
    )
    for index, (label, path, is_mask) in enumerate(panels):
        panel = (
            Image.open(path)
            .convert("RGB")
            .resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        )
        left = index * panel_width
        sheet.paste(panel, (left, header_height))
        draw.rectangle((left, 80, left + panel_width, 112), fill="#1c2028")
        detail = f"  support {mask_coverage(path):.1f}%" if is_mask else ""
        draw.text((left + 10, 86), label + detail, fill="white", font=label_font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def compose_before_after_sheet(
    *,
    title: str,
    subtitle: str,
    image_path: Path,
    before_path: Path,
    after_path: Path,
    destination: Path,
) -> None:
    """Write a labeled source, before, and after regression sheet."""

    panels = (
        ("GENERATED IMAGE", image_path, False),
        ("BEFORE", before_path, True),
        ("CORRECTED DEFAULT", after_path, True),
    )
    panel_width = 360
    panel_height = 480
    header_height = 112
    sheet = Image.new(
        "RGB", (panel_width * len(panels), header_height + panel_height), "#101216"
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 12), title, fill="white", font=load_font(30))
    draw.text((18, 54), subtitle, fill="#b9c1cc", font=load_font(15))
    for index, (label, path, is_mask) in enumerate(panels):
        panel = (
            Image.open(path)
            .convert("RGB")
            .resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        )
        left = index * panel_width
        sheet.paste(panel, (left, header_height))
        draw.rectangle((left, 80, left + panel_width, 112), fill="#1c2028")
        detail = f"  support {mask_coverage(path):.1f}%" if is_mask else ""
        draw.text((left + 10, 86), label + detail, fill="white", font=load_font(18))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def mask_coverage(path: Path) -> float:
    """Return the percentage of pixels with visible mask support."""

    values = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return float((values >= 8).mean() * 100.0)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a stable Windows UI font with a portable fallback."""

    path = Path("C:/Windows/Fonts/segoeui.ttf")
    return ImageFont.truetype(path, size) if path.exists() else ImageFont.load_default()
