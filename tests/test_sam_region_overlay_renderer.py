# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for deterministic SAM-style region overlay rendering."""

from __future__ import annotations

import torch

from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment
from simple_syrup.runtime.sam_region_overlay_renderer import SAMRegionOverlayRenderer


def test_renderer_colors_retained_regions_and_preserves_uncovered_pixels() -> None:
    """The overlay colors each SEG while leaving the source visible elsewhere."""

    image = torch.full((1, 8, 10, 3), 0.25)
    first = _segment(CropRegion(1, 1, 6, 6), torch.ones((5, 5)), "first")
    second_mask = torch.zeros((5, 5))
    second_mask[1:4, 1:4] = 1.0
    second = _segment(CropRegion(4, 2, 9, 7), second_mask, "second")

    overlay = SAMRegionOverlayRenderer().render(
        image=image,
        segs=((8, 10), (first, second)),
    )

    assert overlay.shape == image.shape
    assert overlay.dtype == image.dtype
    assert torch.equal(overlay[:, 0, 0], image[:, 0, 0])
    assert not torch.equal(overlay[:, 2, 2], image[:, 2, 2])
    assert not torch.equal(overlay[:, 4, 5], overlay[:, 2, 2])
    assert torch.equal(image, torch.full_like(image, 0.25))


def test_renderer_is_deterministic_and_empty_segs_return_source_copy() -> None:
    """Stable colors support comparisons and empty detections remain readable."""

    image = torch.rand((1, 6, 6, 3), generator=torch.Generator().manual_seed(4))
    segment = _segment(CropRegion(1, 1, 5, 5), torch.ones((4, 4)), "subject")
    renderer = SAMRegionOverlayRenderer()

    first = renderer.render(image=image, segs=((6, 6), (segment,)))
    second = renderer.render(image=image, segs=((6, 6), (segment,)))
    empty = renderer.render(image=image, segs=((6, 6), ()))

    assert torch.equal(first, second)
    assert torch.equal(empty, image)
    assert empty.data_ptr() != image.data_ptr()


def _segment(region: CropRegion, mask: torch.Tensor, label: str) -> Segment:
    """Create one source-aligned segment for renderer tests."""

    return Segment(
        cropped_image=None,
        cropped_mask=mask,
        confidence=1.0,
        crop_region=region,
        bbox=BoundingBox(*region),
        label=label,
    )
