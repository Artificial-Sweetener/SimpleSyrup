# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for external LLM SEG crop image encoding."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import cast

import pytest
import torch
from PIL import Image

from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment
from simple_syrup.runtime.external_llm_images import ExternalLLMSegsImageEncoder


def test_segs_image_encoder_returns_transparent_mask_png() -> None:
    """Transparent mode hides outside-mask pixels with PNG alpha."""

    segment = _segment(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    encoded = ExternalLLMSegsImageEncoder().encode_segment_as_data_url(
        _image(),
        segment,
        "transparent mask",
    )

    image = _decode_png(encoded)
    assert image.mode == "RGBA"
    assert image.size == (2, 2)
    assert _rgba_pixel(image, 0, 0)[3] == 255
    assert _rgba_pixel(image, 1, 0)[3] == 0
    assert _rgba_pixel(image, 0, 1)[3] == 0
    assert _rgba_pixel(image, 1, 1)[3] == 255


def test_segs_image_encoder_returns_black_mask_png() -> None:
    """Black mode zeros outside-mask pixels and keeps RGB output."""

    segment = _segment(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    encoded = ExternalLLMSegsImageEncoder().encode_segment_as_data_url(
        _image(),
        segment,
        "black mask",
    )

    image = _decode_png(encoded)
    assert image.mode == "RGB"
    assert image.size == (2, 2)
    assert image.getpixel((1, 0)) == (0, 0, 0)
    assert image.getpixel((0, 1)) == (0, 0, 0)
    assert image.getpixel((0, 0)) != (0, 0, 0)
    assert image.getpixel((1, 1)) != (0, 0, 0)


def test_segs_image_encoder_returns_full_crop_png() -> None:
    """Full crop mode preserves the whole crop rectangle."""

    segment = _segment(torch.zeros((2, 2)))

    encoded = ExternalLLMSegsImageEncoder().encode_segment_as_data_url(
        _image(),
        segment,
        "full crop",
    )

    image = _decode_png(encoded)
    assert image.mode == "RGB"
    assert image.size == (2, 2)
    assert image.getpixel((0, 0)) != (0, 0, 0)
    assert image.getpixel((1, 0)) != (0, 0, 0)
    assert image.getpixel((0, 1)) != (0, 0, 0)
    assert image.getpixel((1, 1)) != (0, 0, 0)


def test_segs_image_encoder_accepts_full_image_masks() -> None:
    """Full-image SEG masks are cropped to the SEG crop region."""

    full_mask = torch.zeros((4, 4), dtype=torch.float32)
    full_mask[1, 1] = 1.0
    full_mask[2, 2] = 1.0
    segment = _segment(full_mask)

    encoded = ExternalLLMSegsImageEncoder().encode_segment_as_data_url(
        _image(),
        segment,
        "transparent mask",
    )

    image = _decode_png(encoded)
    assert [_rgba_pixel(image, x, y)[3] for y in range(2) for x in range(2)] == [
        255,
        0,
        0,
        255,
    ]


def test_segs_image_encoder_rejects_invalid_masks() -> None:
    """SEG masks must be HW or BHW tensors."""

    segment = _segment(torch.zeros((1, 1, 1, 1), dtype=torch.float32))

    with pytest.raises(ValueError, match="HW or BHW"):
        ExternalLLMSegsImageEncoder().encode_segment_as_data_url(
            _image(),
            segment,
            "transparent mask",
        )


def test_segs_image_encoder_rejects_unknown_mode() -> None:
    """SEG image mode is restricted to the node combo choices."""

    with pytest.raises(ValueError, match="seg_image_mode"):
        ExternalLLMSegsImageEncoder().encode_segment_as_data_url(
            _image(),
            _segment(torch.ones((2, 2), dtype=torch.float32)),
            "white mask",
        )


def _decode_png(data_url: str) -> Image.Image:
    """Decode a PNG data URL into a PIL image."""

    assert data_url.startswith("data:image/png;base64,")
    payload = data_url.removeprefix("data:image/png;base64,")
    return Image.open(BytesIO(base64.b64decode(payload)))


def _rgba_pixel(image: Image.Image, x: int, y: int) -> tuple[int, int, int, int]:
    """Return one RGBA pixel with a precise test type."""

    return cast(tuple[int, int, int, int], image.getpixel((x, y)))


def _segment(mask: torch.Tensor) -> Segment:
    """Return a SEG that crops a stable two-by-two image region."""

    return Segment(
        cropped_image=None,
        cropped_mask=mask,
        confidence=1.0,
        crop_region=CropRegion(1, 1, 3, 3),
        bbox=BoundingBox(1, 1, 3, 3),
        label="seg",
    )


def _image() -> torch.Tensor:
    """Return a deterministic BHWC image with no black crop pixels."""

    return (
        torch.arange(1, 4 * 4 * 3 + 1, dtype=torch.float32).reshape(1, 4, 4, 3) / 255.0
    )
