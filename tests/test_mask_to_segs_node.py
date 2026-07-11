# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Mask to SEGS Comfy v3 node."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest
import torch

from simple_syrup.domain.segs import BoundingBox, CropRegion, NativeSegs, Segment
from simple_syrup.nodes_v3.mask_to_segs import MaskToSEGSV3
from simple_syrup.services.segs_output_service import CombinedSegsResult


def test_mask_to_segs_v3_schema() -> None:
    """The v3 schema exposes the image-associated mask conversion contract."""

    schema = MaskToSEGSV3.define_schema()

    assert schema.node_id == "SimpleSyrup.MaskToSEGS"
    assert schema.display_name == "Mask to SEGS"
    assert schema.category == "SimpleSyrup/Detection"
    assert [input_item.id for input_item in schema.inputs] == [
        "image",
        "mask",
        "mask_threshold",
        "size_threshold",
        "keep_only",
        "mask_dilation",
        "post_dilation",
        "crop_factor",
        "sort_order",
        "combine_segs",
        "label",
    ]
    assert "confidence_threshold" not in {input_item.id for input_item in schema.inputs}
    assert "detector_model" not in {input_item.id for input_item in schema.inputs}
    assert "keep_by" not in {input_item.id for input_item in schema.inputs}
    assert [input_item.io_type for input_item in schema.inputs[:2]] == [
        "IMAGE",
        "MASK",
    ]
    assert schema.inputs[2].default == 0.5
    assert schema.inputs[3].default == 10
    assert schema.inputs[4].default == 0
    assert schema.inputs[5].default == 0
    assert schema.inputs[6].default == 0
    assert schema.inputs[7].default == 3.0
    assert schema.inputs[8].default == "largest to smallest"
    assert schema.inputs[9].default is False
    assert schema.inputs[10].default == "mask"
    assert [output.id for output in schema.outputs] == ["segs", "mask"]
    assert [output.io_type for output in schema.outputs] == ["SEGS", "MASK"]
    assert schema.outputs[0].is_output_list is True


def test_mask_to_segs_execute_returns_separate_segs(monkeypatch: Any) -> None:
    """Execution returns limited and sorted separate SEGS when combine is false."""

    _FakeMaskService.responses = [_ranking_segs()]
    _FakeMaskService.calls = []
    builder = _RecordingBuilder()
    monkeypatch.setattr(MaskToSEGSV3, "service_class", _FakeMaskService)
    monkeypatch.setattr(MaskToSEGSV3, "combined_builder", builder)

    segs, mask = MaskToSEGSV3.execute(
        image=torch.zeros((1, 8, 8, 3), dtype=torch.float32),
        mask=torch.zeros((1, 8, 8), dtype=torch.float32),
        mask_threshold=0.5,
        size_threshold=1,
        keep_only=2,
        mask_dilation=3,
        post_dilation=-1,
        crop_factor=1.0,
        sort_order="left to right",
        combine_segs=False,
        label="paint",
    )

    segs_list = cast(list[tuple[object, list[Segment]]], segs)
    _header, segments = segs_list[0]
    assert [segment.label for segment in segments] == ["medium", "large-low"]
    assert builder.seen_labels == [["medium", "large-low"]]
    assert mask.shape == (1, 8, 8)
    assert _FakeMaskService.calls[0]["mask_threshold"] == 0.5
    assert _FakeMaskService.calls[0]["mask_dilation"] == 3
    assert _FakeMaskService.calls[0]["post_dilation"] == -1
    assert _FakeMaskService.calls[0]["label"] == "paint"


def test_mask_to_segs_execute_returns_combined_segs(monkeypatch: Any) -> None:
    """Execution returns one combined SEG when combine is enabled."""

    _FakeMaskService.responses = [_ranking_segs()]
    _FakeMaskService.calls = []
    builder = _RecordingBuilder()
    monkeypatch.setattr(MaskToSEGSV3, "service_class", _FakeMaskService)
    monkeypatch.setattr(MaskToSEGSV3, "combined_builder", builder)

    segs, mask = MaskToSEGSV3.execute(
        image=torch.zeros((1, 8, 8, 3), dtype=torch.float32),
        mask=torch.zeros((8, 8), dtype=torch.float32),
        mask_threshold=0.5,
        size_threshold=1,
        keep_only=1,
        mask_dilation=0,
        post_dilation=0,
        crop_factor=1.0,
        sort_order="largest to smallest",
        combine_segs=True,
        label="paint",
    )

    segs_list = cast(list[tuple[object, list[Segment]]], segs)
    _header, segments = segs_list[0]
    assert [segment.label for segment in segments] == ["combined"]
    assert builder.seen_labels == [["large-low"]]
    assert torch.equal(mask, torch.ones((1, 8, 8)))


def test_mask_to_segs_execute_reuses_single_mask_for_image_batch(
    monkeypatch: Any,
) -> None:
    """A single mask can be reused for every image in a batch."""

    _FakeMaskService.responses = [
        _segs(_segment("image-0", CropRegion(0, 0, 2, 2))),
        _segs(_segment("image-1", CropRegion(0, 0, 2, 2))),
    ]
    _FakeMaskService.calls = []
    monkeypatch.setattr(MaskToSEGSV3, "service_class", _FakeMaskService)
    monkeypatch.setattr(MaskToSEGSV3, "combined_builder", _RecordingBuilder())

    segs, mask = MaskToSEGSV3.execute(
        image=torch.zeros((2, 8, 8, 3), dtype=torch.float32),
        mask=torch.zeros((1, 8, 8), dtype=torch.float32),
        mask_threshold=0.5,
        size_threshold=1,
        keep_only=0,
        mask_dilation=0,
        post_dilation=0,
        crop_factor=1.0,
        sort_order="largest to smallest",
        combine_segs=False,
        label="paint",
    )

    segs_list = cast(list[tuple[object, list[Segment]]], segs)
    assert [segments[0].label for _header, segments in segs_list] == [
        "image-0",
        "image-1",
    ]
    assert len(_FakeMaskService.calls) == 2
    assert mask.shape == (2, 8, 8)


def test_mask_to_segs_execute_rejects_mismatched_mask_batch() -> None:
    """Mask batches must be singleton or aligned to the image batch."""

    with pytest.raises(ValueError, match="mask batch size"):
        MaskToSEGSV3.execute(
            image=torch.zeros((2, 8, 8, 3), dtype=torch.float32),
            mask=torch.zeros((3, 8, 8), dtype=torch.float32),
            mask_threshold=0.5,
            size_threshold=1,
            keep_only=0,
            mask_dilation=0,
            post_dilation=0,
            crop_factor=1.0,
            sort_order="largest to smallest",
            combine_segs=False,
            label="paint",
        )


class _FakeMaskService:
    """Mask-to-SEGS service double for node tests."""

    responses: ClassVar[list[NativeSegs]] = []
    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self) -> None:
        """Create an empty per-instance response cursor."""

        self._index = 0

    def build(self, **kwargs: object) -> NativeSegs:
        """Return the next configured SEGS payload and record inputs."""

        type(self).calls.append(kwargs)
        response = type(self).responses[self._index]
        self._index += 1
        return response


class _RecordingBuilder:
    """Combined-result builder that records input SEGS order."""

    def __init__(self) -> None:
        """Create empty call records."""

        self.seen_labels: list[list[str]] = []

    def __call__(
        self,
        image: object,
        segs: NativeSegs,
        crop_factor: float,
    ) -> CombinedSegsResult:
        """Return a deterministic combined result and record source labels."""

        del image, crop_factor
        header, segments = segs
        self.seen_labels.append([segment.label for segment in segments])
        return CombinedSegsResult(
            segs=(header, (_segment("combined", CropRegion(0, 0, 8, 8)),)),
            mask=torch.ones((1, header[0], header[1]), dtype=torch.float32),
        )


def _ranking_segs() -> NativeSegs:
    """Return SEGS with rankable area values."""

    return _segs(
        _segment("small-high", CropRegion(0, 0, 2, 2)),
        _segment("large-low", CropRegion(4, 0, 8, 4)),
        _segment("medium", CropRegion(2, 0, 5, 3)),
    )


def _segs(*segments: Segment) -> NativeSegs:
    """Create native test SEGS."""

    return (8, 8), tuple(segments)


def _segment(label: str, crop_region: CropRegion) -> Segment:
    """Create a test segment."""

    return Segment(
        cropped_image=None,
        cropped_mask=torch.ones((crop_region.height, crop_region.width)),
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
