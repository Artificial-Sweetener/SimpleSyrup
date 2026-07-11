# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for shared SEGS output finalization."""

from __future__ import annotations

import torch

from simple_syrup.domain.segs import BoundingBox, CropRegion, NativeSegs, Segment
from simple_syrup.services.segs_output_service import (
    CombinedSegsResult,
    finalize_detector_segs_output,
)


def test_finalization_limits_then_sorts_before_combined_builder() -> None:
    """The shared pipeline limits SEGS before applying final output ordering."""

    builder = _RecordingBuilder()
    output = finalize_detector_segs_output(
        image=torch.zeros((1, 8, 8, 3), dtype=torch.float32),
        segs=_ranking_segs(),
        keep_only=2,
        keep_by="largest size",
        crop_factor=1.5,
        sort_order="left to right",
        combine_segs=False,
        combined_builder=builder,
    )

    _header, segments = output.segs
    assert isinstance(segments, list)
    assert [segment.label for segment in segments] == ["medium", "large-low"]
    assert builder.seen_labels == [["medium", "large-low"]]
    assert builder.seen_crop_factors == [1.5]
    assert torch.equal(output.mask, torch.ones((1, 8, 8), dtype=torch.float32))


def test_finalization_returns_combined_segs_when_requested() -> None:
    """Combined mode emits the unioned SEGS while keeping the same output mask."""

    builder = _RecordingBuilder()
    output = finalize_detector_segs_output(
        image=torch.zeros((1, 8, 8, 3), dtype=torch.float32),
        segs=_ranking_segs(),
        keep_only=1,
        keep_by="highest confidence",
        crop_factor=1.0,
        sort_order="largest to smallest",
        combine_segs=True,
        combined_builder=builder,
    )

    _header, segments = output.segs
    assert [segment.label for segment in segments] == ["combined"]
    assert builder.seen_labels == [["small-high"]]
    assert torch.equal(output.mask, torch.ones((1, 8, 8), dtype=torch.float32))


def test_finalization_supports_largest_size_for_mask_derived_segs() -> None:
    """Mask-derived SEGS can use the existing largest-size ranking policy."""

    builder = _RecordingBuilder()
    output = finalize_detector_segs_output(
        image=torch.zeros((1, 8, 8, 3), dtype=torch.float32),
        segs=_ranking_segs(),
        keep_only=1,
        keep_by="largest size",
        crop_factor=1.0,
        sort_order="largest to smallest",
        combine_segs=False,
        combined_builder=builder,
    )

    _header, segments = output.segs
    assert [segment.label for segment in segments] == ["large-low"]
    assert builder.seen_labels == [["large-low"]]


class _RecordingBuilder:
    """Combined-result builder test double."""

    def __init__(self) -> None:
        """Create empty call records."""

        self.seen_labels: list[list[str]] = []
        self.seen_crop_factors: list[float] = []

    def __call__(
        self,
        image: object,
        segs: NativeSegs,
        crop_factor: float,
    ) -> CombinedSegsResult:
        """Return a deterministic combined output while recording inputs."""

        del image
        self.seen_crop_factors.append(crop_factor)
        self.seen_labels.append([segment.label for segment in segs[1]])
        header, _segments = segs
        return CombinedSegsResult(
            segs=(header, (_segment("combined", CropRegion(0, 0, 8, 8), 1.0),)),
            mask=torch.ones((1, header[0], header[1]), dtype=torch.float32),
        )


def _ranking_segs() -> NativeSegs:
    """Return SEGS with rankable confidence and size differences."""

    return (
        (8, 8),
        (
            _segment("small-high", CropRegion(0, 0, 2, 2), 0.95),
            _segment("large-low", CropRegion(4, 0, 8, 4), 0.1),
            _segment("medium", CropRegion(2, 0, 5, 3), 0.5),
        ),
    )


def _segment(
    label: str,
    crop_region: CropRegion,
    confidence: float,
) -> Segment:
    """Create a test segment with a crop-local mask."""

    return Segment(
        cropped_image=None,
        cropped_mask=torch.ones((crop_region.height, crop_region.width)),
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
