# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Materialize deterministic rectangular benchmark masks for ComfyUI."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .manifest_types import BenchmarkCase, MaskRectangle


class MaskArtifactWriter:
    """Write only uniquely prefixed benchmark files in one Comfy input root."""

    def __init__(self, input_root: Path, filename_prefix: str) -> None:
        """Validate the exact filename prefix owned beneath the input root."""

        if not filename_prefix or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in filename_prefix
        ):
            raise ValueError(
                "Benchmark mask prefix must be lowercase and filename-safe."
            )
        self._input_root = input_root.resolve()
        self._filename_prefix = filename_prefix

    def write_case(
        self,
        case: BenchmarkCase,
        *,
        width: int,
        height: int,
    ) -> tuple[str, ...]:
        """Write one RGB PNG per authored mask and return Comfy-relative names."""

        if width < 1 or height < 1:
            raise ValueError("Benchmark mask dimensions must be positive.")
        self._input_root.mkdir(parents=True, exist_ok=True)
        names = []
        for index, rectangle in enumerate(case.masks):
            filename = (
                f"{self._filename_prefix}__{case.case_id}__region-{index:02d}.png"
            )
            path = self._input_root / filename
            self._write_rectangle(path, rectangle, width=width, height=height)
            names.append(filename)
        return tuple(names)

    @staticmethod
    def _write_rectangle(
        path: Path,
        rectangle: MaskRectangle,
        *,
        width: int,
        height: int,
    ) -> None:
        """Write a binary RGB mask using half-open normalized coordinates."""

        x0 = round(rectangle.x0 * width)
        y0 = round(rectangle.y0 * height)
        x1 = round(rectangle.x1 * width)
        y1 = round(rectangle.y1 * height)
        if x0 >= x1 or y0 >= y1:
            raise ValueError("A normalized benchmark mask collapsed at output size.")
        image = Image.new("RGB", (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(255, 255, 255))
        image.save(path, format="PNG", optimize=False, compress_level=9)
