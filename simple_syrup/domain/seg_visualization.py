# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plan deterministic visual representations of validated SEGS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from .segs import CropRegion, NativeSegs, Segment, coerce_segment_mask, coerce_segs


class RegionColor(NamedTuple):
    """Represent one reusable RGB region color."""

    red: int
    green: int
    blue: int

    @property
    def normalized(self) -> tuple[float, float, float]:
        """Return the color as normalized tensor-ready channels."""

        return self.red / 255.0, self.green / 255.0, self.blue / 255.0

    @property
    def css(self) -> str:
        """Return the color as a browser-ready hexadecimal value."""

        return f"#{self.red:02x}{self.green:02x}{self.blue:02x}"


REGION_COLORS: tuple[RegionColor, ...] = (
    RegionColor(242, 66, 54),
    RegionColor(33, 150, 243),
    RegionColor(76, 176, 80),
    RegionColor(255, 194, 8),
    RegionColor(156, 39, 176),
    RegionColor(255, 87, 34),
    RegionColor(0, 188, 212),
    RegionColor(232, 31, 99),
    RegionColor(140, 194, 74),
    RegionColor(103, 58, 183),
    RegionColor(255, 153, 0),
    RegionColor(0, 150, 136),
)


@dataclass(frozen=True)
class VisualRegion:
    """Describe one validated SEG with stable visualization identity."""

    region_id: str
    index: int
    segment: Segment
    mask: torch.Tensor
    color: RegionColor
    active_area: int

    @property
    def crop_region(self) -> CropRegion:
        """Return the source crop occupied by this region."""

        return self.segment.crop_region


@dataclass(frozen=True)
class SegVisualizationPlan:
    """Own source geometry and ordered visual regions for one SEGS payload."""

    source_height: int
    source_width: int
    regions: tuple[VisualRegion, ...]


def build_seg_visualization_plan(segs: NativeSegs) -> SegVisualizationPlan:
    """Return deterministic validated regions without mutating the source SEGS."""

    (source_height, source_width), segments = coerce_segs(segs)
    regions: list[VisualRegion] = []
    for index, segment in enumerate(segments):
        _validate_crop_bounds(
            segment.crop_region,
            source_height=source_height,
            source_width=source_width,
        )
        mask = coerce_segment_mask(segment).detach().cpu()
        regions.append(
            VisualRegion(
                region_id=f"seg-{index + 1:04d}",
                index=index,
                segment=segment,
                mask=mask,
                color=REGION_COLORS[index % len(REGION_COLORS)],
                active_area=int((mask >= 0.5).sum().item()),
            )
        )
    return SegVisualizationPlan(
        source_height=source_height,
        source_width=source_width,
        regions=tuple(regions),
    )


def _validate_crop_bounds(
    crop: CropRegion,
    *,
    source_height: int,
    source_width: int,
) -> None:
    """Reject SEG crops that cannot describe the declared source image."""

    if crop.right > source_width or crop.bottom > source_height:
        raise ValueError("Segment crop_region must fit inside the SEGS dimensions.")
