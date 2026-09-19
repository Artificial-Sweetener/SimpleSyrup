# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persist decoded NegPiP images and assemble visible paired proof."""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import cast

from PIL import Image, ImageChops, ImageDraw, ImageStat

from tools.comfy_api import JsonObject
from tools.comfy_integration.portable_font import load_label_font

from .workflow import NEGATIVE_WEIGHT_LABEL, NegpipLiveFamily

EXPECTED_IMAGE_SIZE = (512, 512)


class NegpipVisualProofRecorder:
    """Own original decoded images, pixel comparisons, and a contact sheet."""

    def __init__(self, root: Path) -> None:
        """Retain one existing managed artifact directory."""

        self._root = root.resolve()

    def record(
        self,
        family: NegpipLiveFamily,
        mode: str,
        image_bytes: bytes,
    ) -> JsonObject:
        """Validate and persist one original Comfy PNG."""

        if mode not in {"control", "negative", "ppm_baseline"}:
            raise ValueError(f"Unknown NegPiP visual proof mode: {mode!r}.")
        if not image_bytes:
            raise ValueError("NegPiP visual proof image is empty.")
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            decoded.load()
            if decoded.format != "PNG":
                raise ValueError("NegPiP visual proof output must be a PNG.")
            image = decoded.convert("RGB")
        if image.size != EXPECTED_IMAGE_SIZE:
            raise ValueError(
                "NegPiP visual proof image must be 512x512; "
                f"received {image.size[0]}x{image.size[1]}."
            )
        path = self._root / f"{family.value}-{mode}.png"
        path.write_bytes(image_bytes)
        pixels = image.tobytes()
        return {
            "file": path.name,
            "png_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "rgb_sha256": hashlib.sha256(pixels).hexdigest(),
            "size_bytes": len(image_bytes),
            "width": image.width,
            "height": image.height,
            "rgb_dynamic_range": max(pixels) - min(pixels),
        }

    def finalize(self, families: JsonObject) -> JsonObject:
        """Require every pair, measure its pixels, and create a labeled sheet."""

        expected = {family.value for family in NegpipLiveFamily}
        if set(families) != expected:
            raise ValueError("NegPiP visible proof does not cover every model family.")
        for family in NegpipLiveFamily:
            family_result = _mapping(families.get(family.value), family.value)
            control = _mapping(family_result.get("control"), "control")
            negative = _mapping(family_result.get("negative"), "negative")
            comparison = self._compare(
                _image_path(self._root, control),
                _image_path(self._root, negative),
            )
            if comparison["changed_pixels"] == 0:
                raise ValueError(f"{family.value} control and NegPiP images are equal.")
            family_result["image_comparison"] = comparison
            ppm_baseline = family_result.get("ppm_baseline")
            if ppm_baseline is not None:
                baseline = _mapping(ppm_baseline, "ppm_baseline")
                family_result["ppm_automatic_comparison"] = self._compare(
                    _image_path(self._root, baseline),
                    _image_path(self._root, negative),
                )
        sheet = self._contact_sheet(families)
        data = sheet.read_bytes()
        with Image.open(io.BytesIO(data)) as decoded:
            width, height = decoded.size
        return {
            "file": sheet.name,
            "png_sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "width": width,
            "height": height,
        }

    @staticmethod
    def _compare(control_path: Path, negative_path: Path) -> JsonObject:
        """Return exact RGB difference evidence for a matched-seed pair."""

        with Image.open(control_path) as control_source:
            control = control_source.convert("RGB")
        with Image.open(negative_path) as negative_source:
            negative = negative_source.convert("RGB")
        if control.size != negative.size:
            raise ValueError("NegPiP visual proof pair dimensions do not match.")
        difference = ImageChops.difference(control, negative)
        difference_bytes = difference.tobytes()
        element_count = len(difference_bytes)
        changed_pixels = sum(
            any(difference_bytes[index : index + 3])
            for index in range(0, element_count, 3)
        )
        absolute_sum = sum(difference_bytes)
        squared_sum = sum(value * value for value in difference_bytes)
        ordered_deltas = sorted(difference_bytes)
        p99_index = math.ceil(len(ordered_deltas) * 0.99) - 1
        stat = ImageStat.Stat(difference)
        return {
            "changed_pixels": changed_pixels,
            "total_pixels": control.width * control.height,
            "mean_absolute_rgb_delta": absolute_sum / element_count,
            "root_mean_square_rgb_delta": math.sqrt(squared_sum / element_count),
            "channel_mean_absolute_delta": list(stat.mean),
            "maximum_channel_delta": max(difference_bytes, default=0),
            "p99_channel_delta": ordered_deltas[p99_index],
            "difference_bbox": list(difference.getbbox() or (0, 0, 0, 0)),
        }

    def _contact_sheet(self, families: JsonObject) -> Path:
        """Render original images and runtime observations into one proof sheet."""

        margin = 28
        label_width = 230
        image_size = 340
        column_gap = 20
        header_height = 150
        row_height = 470
        width = margin * 2 + label_width + image_size * 3 + column_gap * 2
        height = header_height + row_height * len(NegpipLiveFamily) + margin
        canvas = Image.new("RGB", (width, height), (18, 20, 24))
        draw = ImageDraw.Draw(canvas)
        title_font = load_label_font(30)
        heading_font = load_label_font(22)
        detail_font = load_label_font(16)
        draw.text(
            (margin, 20),
            "Automatic NegPiP - decoded ComfyUI proof",
            fill="white",
            font=title_font,
        )
        draw.text(
            (margin, 62),
            (f"Same seed 4,205,191 | 512x512 | target: {NEGATIVE_WEIGHT_LABEL}"),
            fill=(198, 204, 214),
            font=detail_font,
        )
        control_x = margin + label_width
        baseline_x = control_x + image_size + column_gap
        negative_x = baseline_x + image_size + column_gap
        draw.text(
            (control_x, 102),
            "CONTROL (+1)",
            fill=(120, 205, 255),
            font=heading_font,
        )
        draw.text(
            (baseline_x, 102),
            "PINNED PPM (-1)",
            fill=(255, 170, 120),
            font=heading_font,
        )
        draw.text(
            (negative_x, 102),
            "SIMPLESYRUP AUTO (-1)",
            fill=(170, 235, 165),
            font=heading_font,
        )
        for row, family in enumerate(NegpipLiveFamily):
            top = header_height + row * row_height
            family_result = _mapping(families.get(family.value), family.value)
            control = _mapping(family_result.get("control"), "control")
            negative = _mapping(family_result.get("negative"), "negative")
            runtime = _mapping(negative.get("runtime"), "runtime")
            comparison = _mapping(
                family_result.get("image_comparison"), "image_comparison"
            )
            with Image.open(_image_path(self._root, control)) as source:
                control_image = source.convert("RGB").resize(
                    (image_size, image_size), Image.Resampling.LANCZOS
                )
            with Image.open(_image_path(self._root, negative)) as source:
                negative_image = source.convert("RGB").resize(
                    (image_size, image_size), Image.Resampling.LANCZOS
                )
            canvas.paste(control_image, (control_x, top))
            ppm_baseline = family_result.get("ppm_baseline")
            if ppm_baseline is None:
                draw.rectangle(
                    (
                        baseline_x,
                        top,
                        baseline_x + image_size,
                        top + image_size,
                    ),
                    fill=(30, 33, 39),
                    outline=(75, 80, 90),
                    width=2,
                )
                draw.multiline_text(
                    (baseline_x + 36, top + 130),
                    "PPM has no Krea 2 path\n\nSimpleSyrup extension test ->",
                    fill=(180, 186, 196),
                    font=detail_font,
                    spacing=8,
                )
            else:
                baseline = _mapping(ppm_baseline, "ppm_baseline")
                with Image.open(_image_path(self._root, baseline)) as source:
                    baseline_image = source.convert("RGB").resize(
                        (image_size, image_size), Image.Resampling.LANCZOS
                    )
                canvas.paste(baseline_image, (baseline_x, top))
            canvas.paste(negative_image, (negative_x, top))
            parity_detail = f"concept: {NEGATIVE_WEIGHT_LABEL}"
            if ppm_baseline is not None:
                parity = _mapping(
                    family_result.get("ppm_automatic_comparison"),
                    "ppm comparison",
                )
                mean_delta = cast(float, parity.get("mean_absolute_rgb_delta"))
                parity_detail = f"PPM~AUTO MAE: {float(mean_delta):.3f}/255"
            draw.text(
                (margin, top + 8),
                family.value.upper().replace("_", " "),
                fill="white",
                font=heading_font,
            )
            draw.multiline_text(
                (margin, top + 52),
                (
                    "control marker: OFF\n"
                    "trigger marker: ON\n"
                    f"callback: {runtime.get('family')}\n"
                    f"calls: {runtime.get('attention_calls')}\n"
                    f"negative signs: {runtime.get('negative_mask_calls')}\n"
                    f"negative tokens: {runtime.get('negative_token_count')}\n"
                    f"control changed: {comparison.get('changed_pixels')}\n"
                    f"{parity_detail}"
                ),
                fill=(200, 206, 216),
                font=detail_font,
                spacing=8,
            )
            draw.line(
                (margin, top + row_height - 18, width - margin, top + row_height - 18),
                fill=(64, 68, 76),
                width=2,
            )
        path = self._root / "negpip-visible-proof.png"
        canvas.save(path, format="PNG")
        return path


def _image_path(root: Path, case: JsonObject) -> Path:
    """Resolve one recorded image inside the artifact root."""

    image = _mapping(case.get("image"), "image")
    relative = image.get("file")
    if not isinstance(relative, str) or not relative:
        raise ValueError("NegPiP visual proof image file is missing.")
    path = (root / relative).resolve()
    if path.parent != root:
        raise ValueError("NegPiP visual proof image escaped its artifact root.")
    if not path.is_file():
        raise FileNotFoundError(f"NegPiP visual proof image is missing: {path}.")
    return path


def _mapping(value: object, field: str) -> JsonObject:
    """Narrow one JSON object with string keys."""

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field} must be a JSON object.")
    return cast(JsonObject, value)
