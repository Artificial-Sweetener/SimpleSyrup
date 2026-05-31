# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Batch SEGS Comfy v3 wrapper."""

from __future__ import annotations

from typing import cast

import pytest

from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment
from simple_syrup.nodes_v3.batch_segs import BatchSEGSV3


def test_batch_segs_v3_schema_uses_autogrow_segs_inputs() -> None:
    """The v3 schema exposes expandable SEGS inputs."""

    schema = BatchSEGSV3.define_schema()

    assert schema.node_id == "SimpleSyrup.BatchSEGS"
    assert schema.display_name == "Batch SEGS"
    assert schema.category == "SimpleSyrup/Detection"
    assert [input_item.id for input_item in schema.inputs] == ["segs_inputs"]
    assert schema.inputs[0].io_type == "COMFY_AUTOGROW_V3"
    assert schema.inputs[0].template.prefix == "segs"
    assert schema.inputs[0].template.min == 2
    assert schema.inputs[0].template.max == 50
    assert schema.inputs[0].template.input.io_type == "SEGS"
    assert [output.id for output in schema.outputs] == ["segs"]
    assert schema.outputs[0].io_type == "SEGS"


def test_batch_segs_v3_execute_returns_impact_compatible_segs() -> None:
    """The v3 wrapper batches SEGS in Autogrow insertion order."""

    first = (
        (16, 16),
        [
            _segment("1", CropRegion(0, 0, 2, 2)),
            _segment("2", CropRegion(2, 0, 4, 2)),
            _segment("3", CropRegion(4, 0, 6, 2)),
        ],
    )
    second = (
        (16, 16),
        (
            _segment("4", CropRegion(0, 2, 2, 4)),
            _segment("5", CropRegion(2, 2, 4, 4)),
            _segment("6", CropRegion(4, 2, 6, 4)),
        ),
    )

    (raw_segs,) = BatchSEGSV3.execute({"segs0": first, "segs1": second})
    segs = cast(tuple[tuple[int, int], list[Segment]], raw_segs)

    header, segments = segs
    assert header == (16, 16)
    assert isinstance(segments, list)
    assert [segment.label for segment in segments] == ["1", "2", "3", "4", "5", "6"]


def test_batch_segs_v3_execute_surfaces_header_mismatch() -> None:
    """The v3 wrapper keeps domain validation errors visible."""

    first = ((8, 16), (_segment("first", CropRegion(0, 0, 2, 2)),))
    second = ((16, 8), (_segment("second", CropRegion(0, 0, 2, 2)),))

    with pytest.raises(ValueError, match="input 2 is 16x8 but input 1 is 8x16"):
        BatchSEGSV3.execute({"segs0": first, "segs1": second})


def _segment(label: str, crop_region: CropRegion) -> Segment:
    """Create a segment for Batch SEGS v3 tests."""

    return Segment(
        cropped_image=None,
        cropped_mask="mask",
        confidence=1.0,
        crop_region=crop_region,
        bbox=BoundingBox(
            crop_region.left,
            crop_region.top,
            crop_region.right,
            crop_region.bottom,
        ),
        label=label,
    )
