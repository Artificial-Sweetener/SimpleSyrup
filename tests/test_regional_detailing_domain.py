# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for regional detailing domain pairing."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.regional_detailing import pair_segments_with_conditioning
from simple_syrup.domain.segs import BoundingBox, CropRegion, NativeSegs, Segment


def test_exact_segs_and_conditioning_count_succeeds() -> None:
    """Each SEG is paired with the conditioning entry at the same index."""

    first = _segment("first")
    second = _segment("second", CropRegion(4, 0, 8, 4), BoundingBox(5, 1, 7, 3))

    pairs = pair_segments_with_conditioning(
        _segs(first, second),
        ConditioningBatch(("positive 1", "positive 2")),
        image_height=8,
        image_width=8,
    )

    assert [pair.index for pair in pairs] == [0, 1]
    assert [pair.segment.label for pair in pairs] == ["first", "second"]
    assert [pair.positive for pair in pairs] == ["positive 1", "positive 2"]


def test_too_few_conditioning_entries_fail() -> None:
    """Region prompts must match SEGS cardinality exactly."""

    with pytest.raises(ValueError, match="1 conditioning entries for 2 SEGS"):
        pair_segments_with_conditioning(
            _segs(_segment("first"), _segment("second")),
            ConditioningBatch(("positive 1",)),
            image_height=8,
            image_width=8,
        )


def test_too_many_conditioning_entries_fail() -> None:
    """Extra region prompts are rejected instead of ignored."""

    with pytest.raises(ValueError, match="2 conditioning entries for 1 SEGS"):
        pair_segments_with_conditioning(
            _segs(_segment("first")),
            ConditioningBatch(("positive 1", "positive 2")),
            image_height=8,
            image_width=8,
        )


def test_empty_segs_skip_conditioning_validation() -> None:
    """Empty SEGS can return unchanged without a regional batch."""

    pairs = pair_segments_with_conditioning(
        ((8, 8), ()),
        object(),
        image_height=8,
        image_width=8,
    )

    assert pairs == ()


def test_normal_conditioning_batch_fallback_is_not_used() -> None:
    """The older batch select fallback must not hide count mismatches."""

    with pytest.raises(ValueError, match="1 conditioning entries for 2 SEGS"):
        pair_segments_with_conditioning(
            _segs(_segment("first"), _segment("second")),
            ConditioningBatch(("reused by old detailer",)),
            image_height=8,
            image_width=8,
        )


def test_region_positive_must_be_conditioning_batch() -> None:
    """Non-empty SEGS require the explicit batch type."""

    with pytest.raises(TypeError, match="CONDITIONING_BATCH"):
        pair_segments_with_conditioning(
            _segs(_segment("first")),
            "normal conditioning",
            image_height=8,
            image_width=8,
        )


def test_nested_region_positive_batch_fails() -> None:
    """Region batch entries must be normal conditioning values."""

    with pytest.raises(TypeError, match="entry 0"):
        pair_segments_with_conditioning(
            _segs(_segment("first")),
            ConditioningBatch((ConditioningBatch(("nested",)),)),
            image_height=8,
            image_width=8,
        )


def test_segs_header_and_image_dimensions_must_match() -> None:
    """The SEGS header is validated against the current image size."""

    with pytest.raises(ValueError, match="SEGS is 9x8, image is 8x8"):
        pair_segments_with_conditioning(
            ((9, 8), (_segment("first"),)),
            ConditioningBatch(("positive",)),
            image_height=8,
            image_width=8,
        )


def test_segment_crop_must_fit_inside_image() -> None:
    """Segments outside the image fail before mask conversion."""

    with pytest.raises(ValueError, match="crop_region must fit"):
        pair_segments_with_conditioning(
            _segs(_segment("outside", CropRegion(6, 6, 10, 10))),
            ConditioningBatch(("positive",)),
            image_height=8,
            image_width=8,
        )


def _segs(*segments: Segment) -> NativeSegs:
    """Return native SEGS for an 8x8 image."""

    return (8, 8), tuple(segments)


def _segment(
    label: str,
    crop_region: CropRegion | None = None,
    bbox: BoundingBox | None = None,
) -> Segment:
    """Return one valid test segment."""

    resolved_region = crop_region or CropRegion(0, 0, 4, 4)
    return Segment(
        cropped_image=None,
        cropped_mask=torch.ones((resolved_region.height, resolved_region.width)),
        confidence=1.0,
        crop_region=resolved_region,
        bbox=bbox or BoundingBox(1, 1, 3, 3),
        label=label,
    )
