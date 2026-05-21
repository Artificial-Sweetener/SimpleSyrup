# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for SimpleSyrup SEGS domain compatibility."""

from __future__ import annotations

from collections import namedtuple

import pytest

from simple_syrup.domain.segs import (
    SORT_ORDER_OPTIONS,
    BoundingBox,
    CropRegion,
    Segment,
    coerce_segment,
    coerce_segs,
    coerce_segs_group,
    sort_segs,
    to_impact_compatible_segs,
    to_impact_compatible_segs_group,
)

ImpactSegment = namedtuple(
    "ImpactSegment",
    [
        "cropped_image",
        "cropped_mask",
        "confidence",
        "crop_region",
        "bbox",
        "label",
        "control_net_wrapper",
    ],
)


def test_native_segment_round_trips_to_impact_shape() -> None:
    """Native SEGS emits the tuple/list shape expected by Impact-style nodes."""

    segment = Segment(
        cropped_image=None,
        cropped_mask="mask",
        confidence=0.75,
        crop_region=CropRegion(1, 2, 11, 12),
        bbox=BoundingBox(3, 4, 9, 10),
        label="face",
    )

    header, segments = to_impact_compatible_segs(((100, 200), (segment,)))

    assert header == (100, 200)
    assert segments == [segment]
    assert segments[0].crop_region[0] == 1
    assert segments[0].bbox.right == 9


def test_impact_namedtuple_segment_is_accepted() -> None:
    """Attribute-compatible Impact segments coerce to native segments."""

    impact_segment = ImpactSegment(
        None,
        "mask",
        0.5,
        [0, 0, 16, 24],
        [4, 6, 10, 18],
        "person",
        None,
    )

    header, segments = coerce_segs(((24, 16), [impact_segment]))

    assert header == (24, 16)
    assert segments[0] == Segment(
        cropped_image=None,
        cropped_mask="mask",
        confidence=0.5,
        crop_region=CropRegion(0, 0, 16, 24),
        bbox=BoundingBox(4, 6, 10, 18),
        label="person",
    )


def test_missing_segment_attributes_are_rejected() -> None:
    """Invalid segment objects fail with a useful message."""

    with pytest.raises(ValueError, match="missing required attribute"):
        coerce_segment(object())


def test_invalid_crop_coordinates_are_rejected() -> None:
    """Crop coordinates must define a positive region."""

    impact_segment = ImpactSegment(
        None, "mask", 1.0, [5, 0, 5, 10], [0, 0, 1, 1], "x", None
    )

    with pytest.raises(ValueError, match="crop_region"):
        coerce_segment(impact_segment)


def test_empty_segment_list_is_valid() -> None:
    """SEGS can represent no detections."""

    assert coerce_segs(((32, 64), [])) == ((32, 64), ())


def test_single_segs_coerces_to_one_item_group() -> None:
    """A normal SEGS payload is accepted as a one-item group."""

    segment = _segment("face", CropRegion(0, 0, 2, 2), 0.9)

    assert coerce_segs_group(((16, 16), [segment])) == (((16, 16), (segment,)),)


def test_segs_list_coerces_to_group_in_order() -> None:
    """A Comfy list of SEGS becomes an ordered native group."""

    first = ((16, 16), [_segment("first", CropRegion(0, 0, 2, 2), 0.9)])
    second = ((16, 16), [_segment("second", CropRegion(2, 2, 4, 4), 0.8)])

    group = coerce_segs_group([first, second])

    assert [segs[1][0].label for segs in group] == ["first", "second"]


def test_empty_segs_group_is_rejected() -> None:
    """A SEGS group must contain at least one per-image payload."""

    with pytest.raises(ValueError, match="one or more SEGS payloads"):
        coerce_segs_group([])


def test_malformed_segs_group_item_is_rejected_with_index() -> None:
    """Invalid group items fail with an actionable item index."""

    with pytest.raises(ValueError, match="SEGS group item 1 is invalid"):
        coerce_segs_group([("not-a-header", [])])


def test_impact_segs_group_conversion_returns_list_outputs() -> None:
    """SEGS groups convert to Comfy list output values."""

    first = ((16, 16), (_segment("first", CropRegion(0, 0, 2, 2), 0.9),))
    second = ((16, 16), (_segment("second", CropRegion(2, 2, 4, 4), 0.8),))

    output = to_impact_compatible_segs_group((first, second))

    assert isinstance(output, list)
    assert [segments[0].label for _header, segments in output] == ["first", "second"]
    assert all(isinstance(segments, list) for _header, segments in output)


def test_sort_order_options_are_plain_english_and_ordered() -> None:
    """SEGS sort options match the detector node combo contract."""

    assert SORT_ORDER_OPTIONS == (
        "largest to smallest",
        "smallest to largest",
        "widest to thinnest",
        "thinnest to widest",
        "tallest to shortest",
        "shortest to tallest",
        "top to bottom",
        "bottom to top",
        "left to right",
        "right to left",
        "highest confidence first",
        "lowest confidence first",
    )


@pytest.mark.parametrize(
    ("sort_order", "expected_labels"),
    [
        ("largest to smallest", ["large", "wide", "small"]),
        ("smallest to largest", ["small", "wide", "large"]),
        ("widest to thinnest", ["wide", "large", "small"]),
        ("thinnest to widest", ["small", "large", "wide"]),
        ("tallest to shortest", ["large", "small", "wide"]),
        ("shortest to tallest", ["wide", "small", "large"]),
        ("top to bottom", ["large", "wide", "small"]),
        ("bottom to top", ["small", "wide", "large"]),
        ("left to right", ["large", "small", "wide"]),
        ("right to left", ["wide", "small", "large"]),
        ("highest confidence first", ["small", "wide", "large"]),
        ("lowest confidence first", ["large", "wide", "small"]),
    ],
)
def test_sort_segs_orders_by_selected_policy(
    sort_order: str,
    expected_labels: list[str],
) -> None:
    """Each supported sort policy orders by the documented primary key."""

    segs = (
        (16, 16),
        (
            _segment("large", CropRegion(0, 0, 4, 6), 0.2),
            _segment("small", CropRegion(3, 8, 5, 11), 0.9),
            _segment("wide", CropRegion(8, 2, 14, 4), 0.5),
        ),
    )

    _header, sorted_segments = sort_segs(segs, sort_order)

    assert [segment.label for segment in sorted_segments] == expected_labels


def test_sort_segs_uses_confidence_as_first_non_confidence_tie_breaker() -> None:
    """Equal primary values prefer the stronger detection first."""

    lower = _segment("lower", CropRegion(0, 0, 4, 4), 0.4)
    higher = _segment("higher", CropRegion(8, 8, 12, 12), 0.9)

    _header, sorted_segments = sort_segs(
        ((16, 16), (lower, higher)), "largest to smallest"
    )

    assert [segment.label for segment in sorted_segments] == ["higher", "lower"]


def test_sort_segs_confidence_sort_uses_top_left_tie_breakers() -> None:
    """Equal confidence values fall back to top, then left."""

    later = _segment("later", CropRegion(2, 4, 4, 6), 0.8)
    earlier = _segment("earlier", CropRegion(6, 1, 8, 3), 0.8)
    leftmost = _segment("leftmost", CropRegion(1, 1, 3, 3), 0.8)

    _header, sorted_segments = sort_segs(
        ((16, 16), (later, earlier, leftmost)),
        "highest confidence first",
    )

    assert [segment.label for segment in sorted_segments] == [
        "leftmost",
        "earlier",
        "later",
    ]


def test_sort_segs_preserves_original_index_after_all_ties() -> None:
    """Fully tied segments keep their original detection order."""

    first = _segment("first", CropRegion(0, 0, 4, 4), 0.8)
    second = _segment("second", CropRegion(0, 0, 4, 4), 0.8)

    _header, sorted_segments = sort_segs(
        ((16, 16), (first, second)),
        "highest confidence first",
    )

    assert [segment.label for segment in sorted_segments] == ["first", "second"]


def test_sort_segs_rejects_unknown_sort_order() -> None:
    """Unknown sort options fail instead of silently changing behavior."""

    with pytest.raises(ValueError, match="Unknown SEGS sort order"):
        sort_segs(((16, 16), ()), "detection order")


def test_sort_segs_does_not_mutate_input() -> None:
    """Sorting returns a new SEGS tuple without mutating the source order."""

    first = _segment("first", CropRegion(0, 0, 2, 2), 0.1)
    second = _segment("second", CropRegion(0, 0, 4, 4), 0.2)
    segs = ((16, 16), (first, second))

    _header, sorted_segments = sort_segs(segs, "largest to smallest")

    assert [segment.label for segment in sorted_segments] == ["second", "first"]
    assert [segment.label for segment in segs[1]] == ["first", "second"]


def _segment(label: str, crop_region: CropRegion, confidence: float) -> Segment:
    """Create a segment for domain sorting tests."""

    return Segment(
        cropped_image=None,
        cropped_mask="mask",
        confidence=confidence,
        crop_region=crop_region,
        bbox=BoundingBox(
            crop_region.left,
            crop_region.top,
            crop_region.right,
            crop_region.bottom,
        ),
        label=label,
    )
