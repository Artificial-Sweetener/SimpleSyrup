# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for shared SEG visualization planning."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.seg_visualization import build_seg_visualization_plan
from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment


def test_plan_assigns_stable_ids_colors_and_normalized_masks() -> None:
    """The shared plan owns deterministic visualization identity and geometry."""

    first = _segment(CropRegion(1, 2, 4, 4), torch.ones((1, 2, 3)), "face")
    second_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    second = _segment(CropRegion(5, 1, 7, 3), second_mask, "hand")

    plan = build_seg_visualization_plan(((8, 10), (first, second)))

    assert (plan.source_height, plan.source_width) == (8, 10)
    assert [region.region_id for region in plan.regions] == [
        "seg-0001",
        "seg-0002",
    ]
    assert [region.color.css for region in plan.regions] == ["#f24236", "#2196f3"]
    assert plan.regions[0].mask.shape == (2, 3)
    assert plan.regions[1].active_area == 2


def test_plan_rejects_crop_outside_declared_source() -> None:
    """Invalid source geometry fails before any renderer consumes it."""

    segment = _segment(CropRegion(7, 7, 10, 10), torch.ones((3, 3)), "bad")

    with pytest.raises(ValueError, match="fit inside"):
        build_seg_visualization_plan(((8, 8), (segment,)))


def _segment(region: CropRegion, mask: torch.Tensor, label: str) -> Segment:
    """Create one visualization-ready segment."""

    return Segment(
        cropped_image=None,
        cropped_mask=mask,
        confidence=0.75,
        crop_region=region,
        bbox=BoundingBox(*region),
        label=label,
    )
