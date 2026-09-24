# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify combined native-SEGS output construction."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from simple_syrup.domain.segs import BoundingBox, CropRegion, NativeSegs, Segment
from simple_syrup.services.segs_output_service import build_combined_segs_result


def test_empty_segs_build_empty_combined_outputs() -> None:
    """Empty SEGS produce empty combined SEGS and a zero mask."""

    result = build_combined_segs_result(_image(), ((8, 8), ()), crop_factor=1.0)
    assert result.segs == ((8, 8), ())
    assert result.mask.shape == (1, 8, 8)
    assert result.mask.dtype == torch.float32
    assert result.mask.sum().item() == 0.0


def test_combined_result_unions_all_segment_masks() -> None:
    """Combined SEGS contain one unioned segment for all source masks."""

    image = torch.arange(1 * 8 * 8 * 3, dtype=torch.float32).reshape((1, 8, 8, 3))
    image = image / image.max()
    first = _segment(CropRegion(1, 1, 3, 3), 0.4, torch.ones((2, 2)))
    second = _segment(
        CropRegion(4, 5, 7, 7),
        0.9,
        [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
    )
    result = build_combined_segs_result(image, _segs(first, second), crop_factor=1.0)
    _header, segments = result.segs
    assert result.mask.shape == (1, 8, 8)
    assert result.mask.sum().item() == 10.0
    assert len(segments) == 1
    assert segments[0].crop_region == CropRegion(1, 1, 7, 7)
    assert segments[0].bbox == BoundingBox(1, 1, 7, 7)
    assert segments[0].label == "combined"
    assert segments[0].confidence == 0.9
    assert cast(torch.Tensor, segments[0].cropped_mask).shape == (6, 6)
    assert torch.equal(
        cast(torch.Tensor, segments[0].cropped_image), image[:, 1:7, 1:7, :]
    )


def test_combined_result_applies_crop_factor_after_combining() -> None:
    """Combined SEGS expand one unioned target for downstream detailers."""

    image = torch.arange(1 * 8 * 8 * 3, dtype=torch.float32).reshape((1, 8, 8, 3))
    image = image / image.max()
    first_mask = torch.zeros((2, 2), dtype=torch.float32)
    first_mask[0, 0] = 1.0
    second_mask = torch.zeros((2, 2), dtype=torch.float32)
    second_mask[1, 1] = 1.0
    first = Segment(
        cropped_image=None,
        cropped_mask=first_mask,
        confidence=0.8,
        crop_region=CropRegion(2, 2, 4, 4),
        bbox=BoundingBox(2, 2, 3, 3),
        label="face",
    )
    second = Segment(
        cropped_image=None,
        cropped_mask=second_mask,
        confidence=0.9,
        crop_region=CropRegion(4, 4, 6, 6),
        bbox=BoundingBox(5, 5, 6, 6),
        label="face",
    )
    result = build_combined_segs_result(image, _segs(first, second), crop_factor=2.0)
    _header, segments = result.segs
    assert len(segments) == 1
    assert segments[0].crop_region == CropRegion(0, 0, 8, 8)
    assert segments[0].bbox == BoundingBox(2, 2, 6, 6)
    assert cast(torch.Tensor, segments[0].cropped_mask).shape == (8, 8)
    assert torch.equal(cast(torch.Tensor, segments[0].cropped_image), image)


def test_combined_mask_uses_max_for_overlaps() -> None:
    """Overlapping source masks combine by max instead of addition."""

    first = _segment(CropRegion(2, 2, 4, 4), 0.4, torch.full((2, 2), 0.7))
    second = _segment(CropRegion(2, 2, 4, 4), 0.8, torch.full((2, 2), 0.8))
    result = build_combined_segs_result(_image(), _segs(first, second), crop_factor=1.0)
    assert result.mask[0, 2, 2].item() == pytest.approx(0.8)
    assert result.mask.sum().item() == pytest.approx(3.2)


def test_combined_outputs_reject_mismatched_cropped_mask_shape() -> None:
    """Invalid crop-local masks fail before producing misleading outputs."""

    segment = _segment(CropRegion(1, 1, 3, 3), 1.0, torch.ones((3, 3)))
    with pytest.raises(ValueError, match="cropped_mask must match"):
        build_combined_segs_result(_image(), _segs(segment), crop_factor=1.0)


def _segs(*segments: Segment) -> NativeSegs:
    """Return native SEGS for an 8x8 image."""

    return (8, 8), tuple(segments)


def _segment(
    crop_region: CropRegion,
    confidence: float,
    cropped_mask: object,
) -> Segment:
    """Return one segment for combined-output tests."""

    return Segment(
        cropped_image=None,
        cropped_mask=cropped_mask,
        confidence=confidence,
        crop_region=crop_region,
        bbox=BoundingBox(
            crop_region.left,
            crop_region.top,
            crop_region.right,
            crop_region.bottom,
        ),
        label="face",
    )


def _image() -> torch.Tensor:
    """Return a small single-image tensor."""

    return torch.zeros((1, 8, 8, 3), dtype=torch.float32)
