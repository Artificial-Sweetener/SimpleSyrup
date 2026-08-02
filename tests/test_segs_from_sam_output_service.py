# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for automatic SAM output conversion into standard SEGS."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from simple_syrup.runtime.sam_automatic_segmenter import AutomaticSAMMask
from simple_syrup.services.segs_from_sam_output_service import SEGSFromSAMOutputService


class _RecordingSegmenter:
    """Return configured guide-space masks while recording the input shape."""

    def __init__(self, masks: tuple[AutomaticSAMMask, ...]) -> None:
        """Store masks for a deterministic automatic segmentation response."""

        self._masks = masks
        self.image_shapes: list[tuple[int, int]] = []

    def segment_all(
        self,
        sam_model: object,
        image: torch.Tensor,
        execution_device: str = "auto",
    ) -> tuple[AutomaticSAMMask, ...]:
        """Record the guide image and return configured masks."""

        del sam_model, execution_device
        self.image_shapes.append((int(image.shape[1]), int(image.shape[2])))
        return self._masks


class _RecordingPhaseProgress:
    """Record automatic-SEGS phase transitions without a ComfyUI dependency."""

    def __init__(self) -> None:
        """Create an empty phase record."""

        self.phases: list[str] = []

    def advance(self, phase: str) -> None:
        """Record one named phase."""

        self.phases.append(phase)


def test_service_downscales_the_segmentation_guide_without_upscaling_source() -> None:
    """Guide resolution caps the long edge while preserving source SEGS geometry."""

    guide_mask = torch.zeros((32, 64), dtype=torch.float32)
    guide_mask[8:16, 16:32] = 1.0
    runtime = _RecordingSegmenter((AutomaticSAMMask(guide_mask, 0.8),))
    service = SEGSFromSAMOutputService(runtime)

    result = service.build(
        image=torch.zeros((1, 128, 256, 3)),
        sam_model=object(),
        segmentation_resolution=64,
        minimum_region_area=0,
    )

    segs = result.segs
    assert runtime.image_shapes == [(32, 64)]
    assert segs[0] == (128, 256)
    segment = segs[1][0]
    assert segment.bbox == (64, 32, 128, 64)
    assert segment.crop_region == (64, 32, 128, 64)
    assert cast(torch.Tensor, segment.cropped_mask).shape == (32, 64)
    assert segment.confidence == 0.8
    assert segment.label == "segment_001"


def test_service_reports_meaningful_automatic_segmentation_phases() -> None:
    """The service exposes each expensive step to the node-owned progress bar."""

    runtime = _RecordingSegmenter((AutomaticSAMMask(torch.ones((16, 16)), 1.0),))
    reporter = _RecordingPhaseProgress()

    SEGSFromSAMOutputService(runtime).build(
        image=torch.zeros((1, 16, 16, 3)),
        sam_model=object(),
        segmentation_resolution=64,
        minimum_region_area=0,
        phase_progress=reporter,
    )

    assert reporter.phases == [
        "preparing_segmentation_image",
        "generating_automatic_masks",
        "building_segs",
        "rendering_overlay",
    ]


def test_service_filters_region_area_after_restoring_source_dimensions() -> None:
    """Minimum area uses the original image instead of the guide image size."""

    guide_mask = torch.zeros((32, 64), dtype=torch.float32)
    guide_mask[8:10, 8:10] = 1.0
    runtime = _RecordingSegmenter((AutomaticSAMMask(guide_mask, 0.5, "thing"),))
    service = SEGSFromSAMOutputService(runtime)

    retained_result = service.build(
        image=torch.zeros((1, 128, 256, 3)),
        sam_model=object(),
        segmentation_resolution=64,
        minimum_region_area=63,
    )
    filtered_result = service.build(
        image=torch.zeros((1, 128, 256, 3)),
        sam_model=object(),
        segmentation_resolution=64,
        minimum_region_area=65,
    )

    assert retained_result.segs[1][0].label == "thing"
    assert filtered_result.segs[1] == ()


def test_service_suppresses_duplicate_masks_and_keeps_highest_confidence() -> None:
    """Repeated automatic masks do not create duplicate detailer targets."""

    mask = torch.ones((16, 16), dtype=torch.float32)
    runtime = _RecordingSegmenter(
        (
            AutomaticSAMMask(mask, 0.4, "first"),
            AutomaticSAMMask(mask, 0.9, "second"),
        )
    )

    result = SEGSFromSAMOutputService(runtime).build(
        image=torch.zeros((1, 16, 16, 3)),
        sam_model=object(),
        segmentation_resolution=64,
        minimum_region_area=0,
    )

    assert len(result.segs[1]) == 1
    assert result.segs[1][0].confidence == 0.9
    assert result.segs[1][0].label == "second"


def test_service_expands_retained_mask_crops_to_source_resolution() -> None:
    """Source-size SEGS creation never materializes a full-image mask per region."""

    guide_mask = torch.zeros((64, 128), dtype=torch.float32)
    guide_mask[16:32, 32:64] = 1.0

    result = SEGSFromSAMOutputService(
        _RecordingSegmenter((AutomaticSAMMask(guide_mask, 1.0),))
    ).build(
        image=torch.zeros((1, 1024, 2048, 3)),
        sam_model=object(),
        segmentation_resolution=128,
        minimum_region_area=0,
    )

    segment = result.segs[1][0]
    assert segment.crop_region == (512, 256, 1024, 512)
    assert cast(torch.Tensor, segment.cropped_mask).shape == (256, 512)


@pytest.mark.parametrize("resolution", (0, 65, 127))
def test_service_rejects_non_64_step_segmentation_resolution(resolution: int) -> None:
    """Service validation preserves the node's 64-pixel resolution contract."""

    service = SEGSFromSAMOutputService(_RecordingSegmenter(()))

    with pytest.raises(ValueError, match="segmentation_resolution"):
        service.build(
            image=torch.zeros((1, 16, 16, 3)),
            sam_model=object(),
            segmentation_resolution=resolution,
            minimum_region_area=0,
        )
