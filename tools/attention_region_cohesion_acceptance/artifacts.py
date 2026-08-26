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
    default_path: Path,
    solid_path: Path,
    destination: Path,
) -> None:
    """Write a four-panel proof sheet with coverage labels."""

    panels = (
        ("GENERATED IMAGE", image_path, False),
        ("RAW ATTENTION ALPHA", raw_path, True),
        ("DEFAULT COHESIVE", default_path, True),
        ("FULL MATTE SOLIDITY", solid_path, True),
    )
    panel_width = 384
    panel_height = 512
    header_height = 112
    sheet = Image.new(
        "RGB", (panel_width * len(panels), header_height + panel_height), "#101216"
    )
    draw = ImageDraw.Draw(sheet)
    title_font = _font(30)
    label_font = _font(18)
    detail_font = _font(15)
    draw.text((18, 12), title, fill="white", font=title_font)
    draw.text(
        (18, 54),
        f'Concept: "{concept}"  |  Same seed and capture profile across masks',
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
        detail = f"  support {_coverage(path):.1f}%" if is_mask else ""
        draw.text((left + 10, 86), label + detail, fill="white", font=label_font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def _coverage(path: Path) -> float:
    """Return the percentage of pixels with visible mask support."""

    values = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return float((values >= 8).mean() * 100.0)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a stable Windows UI font with a portable fallback."""

    path = Path("C:/Windows/Fonts/segoeui.ttf")
    return ImageFont.truetype(path, size) if path.exists() else ImageFont.load_default()
