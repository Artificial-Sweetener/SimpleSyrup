# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build compact renderer-neutral preview documents from IMAGE and SEGS."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

import torch
import torch.nn.functional as functional

from ..domain.seg_visualization import (
    SegVisualizationPlan,
    build_seg_visualization_plan,
)
from ..domain.segs import CropRegion, NativeSegs
from ..masking.segs_mask_ops import validate_single_image

_MAX_PREVIEW_EDGE = 1024
_ATLAS_PIXEL_BUDGET = 4 * 1024 * 1024
_MAX_ATLAS_EDGE = 2048
_ATLAS_PADDING = 1


@dataclass(frozen=True)
class AtlasPlacement:
    """Locate one region mask inside the packed mask atlas."""

    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class SegPreviewRegion:
    """Describe one interactive region and its packed mask geometry."""

    region_id: str
    index: int
    label: str
    confidence: float
    active_area: int
    color: str
    crop: CropRegion
    atlas: AtlasPlacement


@dataclass(frozen=True)
class SegPreviewDocument:
    """Carry bounded image assets and interaction metadata to the UI adapter."""

    source_width: int
    source_height: int
    preview_width: int
    preview_height: int
    image: torch.Tensor
    atlas: torch.Tensor
    regions: tuple[SegPreviewRegion, ...]


class SimplePreviewSEGSService:
    """Create one compact interactive-preview document without ComfyUI IO."""

    def build(self, *, image: object, segs: NativeSegs) -> SegPreviewDocument:
        """Return a bounded base image, packed masks, and deterministic metadata."""

        source_image = validate_single_image(image, "Simple Preview SEGS")
        plan = build_seg_visualization_plan(segs)
        image_height = int(source_image.shape[1])
        image_width = int(source_image.shape[2])
        if (plan.source_height, plan.source_width) != (image_height, image_width):
            raise ValueError(
                "Simple Preview SEGS requires IMAGE and SEGS dimensions to match."
            )

        preview_width, preview_height = _fit_dimensions(
            image_width,
            image_height,
            _MAX_PREVIEW_EDGE,
        )
        preview_image = _resize_image(
            source_image,
            width=preview_width,
            height=preview_height,
        )
        atlas, regions = _build_atlas(plan)
        return SegPreviewDocument(
            source_width=image_width,
            source_height=image_height,
            preview_width=preview_width,
            preview_height=preview_height,
            image=preview_image,
            atlas=atlas,
            regions=regions,
        )


def _build_atlas(
    plan: SegVisualizationPlan,
) -> tuple[torch.Tensor, tuple[SegPreviewRegion, ...]]:
    """Pack every crop-local mask into one bounded RGB atlas."""

    if not plan.regions:
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32), ()

    total_crop_area = sum(
        region.crop_region.width * region.crop_region.height for region in plan.regions
    )
    preview_scale = min(
        1.0,
        _MAX_PREVIEW_EDGE / max(plan.source_width, plan.source_height),
    )
    budget_scale = sqrt(_ATLAS_PIXEL_BUDGET / max(1, total_crop_area))
    scale = min(preview_scale, budget_scale)

    while True:
        sizes = tuple(
            (
                max(1, round(region.crop_region.width * scale)),
                max(1, round(region.crop_region.height * scale)),
            )
            for region in plan.regions
        )
        placements, atlas_width, atlas_height = _pack_rectangles(sizes)
        if atlas_width <= _MAX_ATLAS_EDGE and atlas_height <= _MAX_ATLAS_EDGE:
            break
        scale *= 0.85
        if scale < 1.0 / max(plan.source_width, plan.source_height):
            raise ValueError("Simple Preview SEGS could not pack the region masks.")

    atlas = torch.zeros((atlas_height, atlas_width), dtype=torch.float32)
    metadata: list[SegPreviewRegion] = []
    for visual_region, placement in zip(plan.regions, placements, strict=True):
        resized = (
            functional.interpolate(
                visual_region.mask.unsqueeze(0).unsqueeze(0).float(),
                size=(placement.height, placement.width),
                mode="nearest-exact",
            )
            .squeeze(0)
            .squeeze(0)
        )
        atlas[
            placement.top : placement.top + placement.height,
            placement.left : placement.left + placement.width,
        ] = resized
        metadata.append(
            SegPreviewRegion(
                region_id=visual_region.region_id,
                index=visual_region.index,
                label=visual_region.segment.label,
                confidence=float(visual_region.segment.confidence),
                active_area=visual_region.active_area,
                color=visual_region.color.css,
                crop=visual_region.crop_region,
                atlas=placement,
            )
        )
    return atlas.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3), tuple(metadata)


def _pack_rectangles(
    sizes: tuple[tuple[int, int], ...],
) -> tuple[tuple[AtlasPlacement, ...], int, int]:
    """Pack ordered rectangles into deterministic height-sorted shelves."""

    padded_area = sum(
        (width + _ATLAS_PADDING) * (height + _ATLAS_PADDING) for width, height in sizes
    )
    widest = max(width for width, _height in sizes)
    shelf_width = min(
        _MAX_ATLAS_EDGE,
        max(widest, ceil(sqrt(max(1, padded_area)))),
    )
    indexed = sorted(
        enumerate(sizes),
        key=lambda item: (-item[1][1], -item[1][0], item[0]),
    )
    placements: list[AtlasPlacement | None] = [None] * len(sizes)
    x = 0
    y = 0
    shelf_height = 0
    used_width = 1
    for index, (width, height) in indexed:
        if x > 0 and x + width > shelf_width:
            y += shelf_height + _ATLAS_PADDING
            x = 0
            shelf_height = 0
        placements[index] = AtlasPlacement(x, y, width, height)
        used_width = max(used_width, x + width)
        x += width + _ATLAS_PADDING
        shelf_height = max(shelf_height, height)
    used_height = max(1, y + shelf_height)
    return (
        tuple(placement for placement in placements if placement is not None),
        used_width,
        used_height,
    )


def _fit_dimensions(width: int, height: int, maximum: int) -> tuple[int, int]:
    """Fit image dimensions inside one maximum edge without upscaling."""

    if max(width, height) <= maximum:
        return width, height
    scale = maximum / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _resize_image(
    image: torch.Tensor,
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    """Resize one BHWC image while preserving its channel count."""

    if image.shape[1:3] == (height, width):
        return image.detach().cpu().clone()
    return (
        functional.interpolate(
            image.detach().cpu().movedim(-1, 1).float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        .movedim(1, -1)
        .clamp(0.0, 1.0)
    )
