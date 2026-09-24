# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for regional detailing mask conversion."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.regional_detailing import SegmentConditioningPair
from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment
from simple_syrup.masking.regional_detailing_masks import (
    build_image_regions,
    build_latent_regions,
    feather_image_mask,
    latent_box_from_mask,
    proportional_latent_box,
    scale_image_regions,
    union_masks,
)


def test_full_image_mask_is_placed_from_crop_local_mask() -> None:
    """Crop-local SEGS masks are pasted at the segment crop region."""

    segment = _segment(CropRegion(2, 1, 6, 5), torch.ones((4, 4)))

    (region,) = build_image_regions(
        (SegmentConditioningPair(0, segment, "positive"),),
        image_height=8,
        image_width=8,
    )

    assert region.image_mask.shape == (8, 8)
    assert torch.all(region.image_mask[1:5, 2:6] == 1.0)
    assert torch.sum(region.image_mask) == 16.0


def test_crop_local_mask_is_resized_to_crop_region_size() -> None:
    """Mask placement uses crop-region dimensions, not raw mask dimensions."""

    segment = _segment(CropRegion(0, 0, 4, 4), torch.ones((2, 2)))

    (region,) = build_image_regions(
        (SegmentConditioningPair(0, segment, "positive"),),
        image_height=8,
        image_width=8,
    )

    assert torch.all(region.image_mask[:4, :4] == 1.0)
    assert torch.sum(region.image_mask) == 16.0


def test_latent_mask_resizes_to_bchw_spatial_dimensions() -> None:
    """Image masks become latent-space masks using the encoded latent shape."""

    image_region = build_image_regions(
        (SegmentConditioningPair(0, _segment(CropRegion(0, 0, 4, 8)), "positive"),),
        image_height=8,
        image_width=8,
    )

    (latent_region,) = build_latent_regions(
        image_region,
        latent_height=4,
        latent_width=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        latent_feather=0,
    )

    assert latent_region.latent_mask.shape == (4, 4)
    assert latent_region.latent_box.x == 0
    assert latent_region.latent_box.width == 2
    assert latent_region.latent_box.height == 4


def test_latent_mask_resizes_to_singleton_depth_spatial_dimensions() -> None:
    """The mask helper uses only final latent height and width values."""

    image_region = build_image_regions(
        (SegmentConditioningPair(0, _segment(CropRegion(4, 0, 8, 8)), "positive"),),
        image_height=8,
        image_width=8,
    )

    (latent_region,) = build_latent_regions(
        image_region,
        latent_height=4,
        latent_width=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        latent_feather=0,
    )

    assert latent_region.latent_mask.shape == (4, 4)
    assert latent_region.latent_box.x == 2
    assert latent_region.latent_box.width == 2


def test_proportional_image_to_latent_box_uses_encoded_shape() -> None:
    """Image coordinates are scaled against actual latent dimensions."""

    box = proportional_latent_box(
        left=2,
        top=4,
        right=6,
        bottom=8,
        image_height=16,
        image_width=8,
        latent_height=8,
        latent_width=4,
    )

    assert (box.x, box.y, box.width, box.height) == (1, 2, 2, 2)


def test_scale_image_regions_resizes_masks_and_crop_regions() -> None:
    """Scaled image regions preserve mask coverage proportionally."""

    regions = build_image_regions(
        (SegmentConditioningPair(0, _segment(CropRegion(0, 0, 2, 2)), "positive"),),
        image_height=4,
        image_width=4,
    )

    (scaled,) = scale_image_regions(regions, image_height=8, image_width=8)

    assert scaled.image_mask.shape == (8, 8)
    assert scaled.crop_region == CropRegion(0, 0, 4, 4)
    assert torch.any(scaled.image_mask[:4, :4] > 0)


def test_empty_latent_region_fails() -> None:
    """Regions with no latent coverage fail closed."""

    with pytest.raises(ValueError, match="produced an empty latent region"):
        latent_box_from_mask(
            torch.zeros((4, 4)),
            region_index=3,
            label="empty",
        )


def test_feather_image_mask_preserves_shape_and_range() -> None:
    """Feathering keeps masks usable for image compositing."""

    mask = torch.zeros((8, 8))
    mask[2:6, 2:6] = 1.0

    feathered = feather_image_mask(mask, 2)

    assert feathered.shape == mask.shape
    assert float(feathered.min()) >= 0.0
    assert float(feathered.max()) <= 1.0


def test_latent_feather_preserves_shape_and_range() -> None:
    """Latent feathering keeps region masks clamped."""

    image_region = build_image_regions(
        (SegmentConditioningPair(0, _segment(CropRegion(0, 0, 8, 8)), "positive"),),
        image_height=8,
        image_width=8,
    )

    (latent_region,) = build_latent_regions(
        image_region,
        latent_height=4,
        latent_width=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        latent_feather=1,
    )

    assert latent_region.latent_mask.shape == (4, 4)
    assert float(latent_region.latent_mask.min()) >= 0.0
    assert float(latent_region.latent_mask.max()) <= 1.0


def test_union_mask_combines_overlaps_without_exceeding_one() -> None:
    """Union masks use max-style composition for overlapping SEGS."""

    first = torch.zeros((4, 4))
    first[:, :3] = 0.75
    second = torch.zeros((4, 4))
    second[:, 1:] = 0.8

    union = union_masks((first, second))

    assert union.shape == (4, 4)
    assert torch.all(union <= 1.0)
    assert torch.allclose(union[:, 1:3], torch.ones((4, 2)) * 0.8)


def _segment(region: CropRegion, mask: object | None = None) -> Segment:
    """Return one segment using a crop-local mask."""

    return Segment(
        cropped_image=None,
        cropped_mask=(
            torch.ones((region.height, region.width)) if mask is None else mask
        ),
        confidence=1.0,
        crop_region=region,
        bbox=BoundingBox(region.left, region.top, region.right, region.bottom),
        label="region",
    )
