# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Batch SEGS legacy node."""

from __future__ import annotations

from typing import cast

from simple_syrup.domain.segs import BoundingBox, CropRegion, ImpactSegs, Segment
from simple_syrup.nodes.batch_segs import BatchSEGS


def test_batch_segs_contract() -> None:
    """Batch SEGS exposes a legacy two-input chainable contract."""

    inputs = BatchSEGS.INPUT_TYPES()

    assert BatchSEGS.RETURN_TYPES == ("SEGS",)
    assert BatchSEGS.RETURN_NAMES == ("segs",)
    assert BatchSEGS.FUNCTION == "batch"
    assert BatchSEGS.CATEGORY == "SimpleSyrup/Detection"
    assert list(inputs["required"]) == ["first", "second"]
    assert inputs["required"]["first"][0] == "SEGS"
    assert inputs["required"]["second"][0] == "SEGS"


def test_batch_segs_node_batches_in_input_order() -> None:
    """The legacy node returns Impact-compatible batched SEGS."""

    first = ((8, 8), [_segment("1"), _segment("2")])
    second = ((8, 8), [_segment("3")])

    (raw_segs,) = BatchSEGS().batch(first, second)
    segs = cast(ImpactSegs, raw_segs)

    _header, segments = segs
    assert isinstance(segments, list)
    assert [segment.label for segment in segments] == ["1", "2", "3"]


def _segment(label: str) -> Segment:
    """Create a small test segment."""

    return Segment(
        cropped_image=None,
        cropped_mask="mask",
        confidence=1.0,
        crop_region=CropRegion(0, 0, 2, 2),
        bbox=BoundingBox(0, 0, 2, 2),
        label=label,
    )
