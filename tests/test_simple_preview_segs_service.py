# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for compact interactive SEGS preview document creation."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment
from simple_syrup.services.simple_preview_segs_service import SimplePreviewSEGSService


def test_service_builds_bounded_image_and_non_overlapping_mask_atlas() -> None:
    """One compact document retains source geometry and every region mask."""

    image = torch.rand((1, 1200, 2400, 3), generator=torch.Generator().manual_seed(8))
    first = _segment(CropRegion(0, 0, 600, 400), "subject")
    second = _segment(CropRegion(1000, 500, 1600, 1100), "clothing")

    document = SimplePreviewSEGSService().build(
        image=image,
        segs=((1200, 2400), (first, second)),
    )

    assert (document.preview_width, document.preview_height) == (1024, 512)
    assert document.image.shape == (1, 512, 1024, 3)
    assert document.atlas.ndim == 4
    assert int(document.atlas.shape[-1]) == 3
    assert [region.label for region in document.regions] == ["subject", "clothing"]
    first_atlas, second_atlas = (region.atlas for region in document.regions)
    assert (
        first_atlas.left + first_atlas.width <= second_atlas.left
        or second_atlas.left + second_atlas.width <= first_atlas.left
        or first_atlas.top + first_atlas.height <= second_atlas.top
        or second_atlas.top + second_atlas.height <= first_atlas.top
    )


def test_service_returns_minimal_assets_for_empty_segs() -> None:
    """Empty SEGS remain inspectable without special frontend transport."""

    document = SimplePreviewSEGSService().build(
        image=torch.zeros((1, 8, 12, 3)),
        segs=((8, 12), ()),
    )

    assert document.image.shape == (1, 8, 12, 3)
    assert document.atlas.shape == (1, 1, 1, 3)
    assert document.regions == ()


def test_service_rejects_image_and_segs_size_mismatch() -> None:
    """Hit testing cannot proceed against ambiguous source coordinates."""

    with pytest.raises(ValueError, match="dimensions to match"):
        SimplePreviewSEGSService().build(
            image=torch.zeros((1, 8, 8, 3)),
            segs=((7, 8), ()),
        )


def _segment(region: CropRegion, label: str) -> Segment:
    """Create one filled region for atlas tests."""

    return Segment(
        cropped_image=None,
        cropped_mask=torch.ones((region.height, region.width)),
        confidence=0.8,
        crop_region=region,
        bbox=BoundingBox(*region),
        label=label,
    )
