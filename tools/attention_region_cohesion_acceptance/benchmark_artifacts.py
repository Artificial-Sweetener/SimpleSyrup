# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compose labeled performance evidence for attention capture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .artifacts import load_font


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Describe the median timings for one model benchmark."""

    label: str
    baseline_seconds: float
    capture_seconds: float
    pixels_identical: bool
    image_path: Path
    mask_path: Path

    @property
    def overhead_seconds(self) -> float:
        """Return absolute attention-capture overhead."""

        return self.capture_seconds - self.baseline_seconds

    @property
    def overhead_percent(self) -> float:
        """Return capture overhead relative to baseline generation."""

        return self.overhead_seconds / self.baseline_seconds * 100


def compose_benchmark_sheet(
    summaries: tuple[BenchmarkSummary, ...],
    destination: Path,
) -> None:
    """Write a compact timing sheet with representative masks."""

    width = 1400
    row_height = 430
    header_height = 110
    sheet = Image.new(
        "RGB", (width, header_height + row_height * len(summaries)), "#101216"
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (24, 18), "Attention Capture Performance", fill="white", font=load_font(32)
    )
    draw.text(
        (24, 62),
        (
            "Matched warmed runs; same seed and one saved image; "
            "server-side time; one concept request"
        ),
        fill="#b9c1cc",
        font=load_font(17),
    )
    for row, summary in enumerate(summaries):
        _draw_summary_row(sheet, draw, row, header_height, row_height, summary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def pixels_identical(first: Path, second: Path) -> bool:
    """Return whether two saved generations contain identical pixels."""

    with Image.open(first) as first_image, Image.open(second) as second_image:
        return bool(
            np.array_equal(
                np.asarray(first_image.convert("RGB")),
                np.asarray(second_image.convert("RGB")),
            )
        )


def _draw_summary_row(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    row: int,
    header_height: int,
    row_height: int,
    summary: BenchmarkSummary,
) -> None:
    """Draw one model's image, mask, and timing comparison."""

    top = header_height + row * row_height
    draw.rectangle((0, top, sheet.width, top + row_height), fill="#151820")
    image = _fit(summary.image_path, (300, 360))
    mask = _fit(summary.mask_path, (300, 360))
    sheet.paste(image, (24, top + 48))
    sheet.paste(mask, (344, top + 48))
    draw.text((24, top + 12), summary.label, fill="white", font=load_font(23))
    x = 690
    draw.text(
        (x, top + 68),
        f"Baseline   {summary.baseline_seconds:.3f} s",
        fill="#d6dae2",
        font=load_font(24),
    )
    draw.text(
        (x, top + 118),
        f"Capture    {summary.capture_seconds:.3f} s",
        fill="#d6dae2",
        font=load_font(24),
    )
    overhead_color = "#8ee6bd" if summary.overhead_percent <= 10 else "#ffd18a"
    draw.text(
        (x, top + 182),
        (
            f"Overhead   {summary.overhead_seconds * 1000:+.0f} ms  "
            f"({summary.overhead_percent:+.1f}%)"
        ),
        fill=overhead_color,
        font=load_font(27),
    )
    equality = (
        "IDENTICAL GENERATED PIXELS" if summary.pixels_identical else "PIXEL MISMATCH"
    )
    equality_color = "#8ee6bd" if summary.pixels_identical else "#ff8f8f"
    draw.text((x, top + 252), equality, fill=equality_color, font=load_font(20))
    draw.text(
        (x, top + 296),
        "Capture records evidence during the original denoising pass.",
        fill="#aeb6c2",
        font=load_font(17),
    )


def _fit(path: Path, size: tuple[int, int]) -> Image.Image:
    """Fit an image inside one fixed proof panel."""

    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        panel = Image.new("RGB", size, "black")
        panel.paste(
            image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
        )
        return panel
