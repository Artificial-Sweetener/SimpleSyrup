# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Domain policy for deterministic tile SEGS construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..masking.segs_mask_ops import (
    crop_region_for_bbox,
    resize_mask,
    validate_single_image,
)
from ..shared.logging import get_logger
from .segs import (
    BoundingBox,
    CropRegion,
    NativeSegs,
    Segment,
)

LOGGER = get_logger(__name__)

IRREGULAR_MASK_MODES: tuple[str, ...] = (
    "Reuse fast",
    "Reuse quality",
    "All random fast",
    "All random quality",
)


@dataclass(frozen=True)
class TileSEGSControls:
    """Validate controls for tile SEGS construction."""

    bbox_size: int
    crop_factor: float
    min_overlap: int
    filter_segs_dilation: int
    mask_irregularity: float
    irregular_mask_mode: str

    def __post_init__(self) -> None:
        """Reject controls that would make tile layout ambiguous."""

        if self.bbox_size < 1:
            raise ValueError("bbox_size must be greater than 0.")
        if self.crop_factor < 1.0:
            raise ValueError("crop_factor must be greater than or equal to 1.0.")
        if self.min_overlap < 0:
            raise ValueError("min_overlap must be greater than or equal to 0.")
        if not 0.0 <= self.mask_irregularity <= 1.0:
            raise ValueError("mask_irregularity must be between 0.0 and 1.0.")
        if self.irregular_mask_mode not in IRREGULAR_MASK_MODES:
            raise ValueError(
                f"irregular_mask_mode must be one of {IRREGULAR_MASK_MODES}."
            )


@dataclass(frozen=True)
class TileRegion:
    """Describe one tile bbox and expanded crop region."""

    bbox: BoundingBox
    crop_region: CropRegion


class TileSEGSBuilder:
    """Build Impact-compatible tile SEGS from an image and optional filters."""

    def build(
        self,
        image: object,
        controls: TileSEGSControls,
    ) -> NativeSegs:
        """Return native tile SEGS in deterministic scan order."""

        image_tensor = validate_single_image(image, "Tile & Tag SEGS")
        height = int(image_tensor.shape[1])
        width = int(image_tensor.shape[2])
        bbox_size, min_overlap = _adjust_tile_controls(width, height, controls)
        tile_regions = _tile_regions(
            width=width,
            height=height,
            bbox_size=bbox_size,
            min_overlap=min_overlap,
            crop_factor=controls.crop_factor,
        )
        mask_factory = _IrregularMaskFactory(controls)
        segments: list[Segment] = []
        for tile_index, tile_region in enumerate(tile_regions):
            mask = _tile_crop_mask(tile_region)
            if controls.mask_irregularity > 0:
                irregular = mask_factory.mask_like(mask, tile_index)
                mask = (mask * irregular).clamp(0.0, 1.0)
            if bool(torch.all(mask == 0.0).item()):
                continue
            label = f"tile_{len(segments) + 1:03}"
            segments.append(
                Segment(
                    cropped_image=None,
                    cropped_mask=mask,
                    confidence=1.0,
                    crop_region=tile_region.crop_region,
                    bbox=tile_region.bbox,
                    label=label,
                )
            )
        return (height, width), tuple(segments)


class _IrregularMaskFactory:
    """Generate deterministic irregular masks for tile blending."""

    def __init__(self, controls: TileSEGSControls) -> None:
        """Initialize the mask generator from node controls."""

        self._controls = controls
        self._reused_noise: torch.Tensor | None = None

    def mask_like(self, reference: torch.Tensor, tile_index: int) -> torch.Tensor:
        """Return a soft irregular mask with the same shape as reference."""

        height = int(reference.shape[0])
        width = int(reference.shape[1])
        if self._controls.irregular_mask_mode.startswith("Reuse"):
            if self._reused_noise is None:
                self._reused_noise = self._noise(height, width, seed=17)
            noise = self._reused_noise
        else:
            noise = self._noise(height, width, seed=101 + tile_index)
        threshold = 1.0 - (self._controls.mask_irregularity * 0.5)
        return torch.where(
            noise > threshold, torch.zeros_like(noise), torch.ones_like(noise)
        )

    def _noise(self, height: int, width: int, seed: int) -> torch.Tensor:
        """Return deterministic low-frequency noise resized to tile shape."""

        generator = torch.Generator(device="cpu").manual_seed(seed)
        quality = 16 if self._controls.irregular_mask_mode.endswith("fast") else 64
        base = torch.rand((quality, quality), generator=generator, dtype=torch.float32)
        resized = resize_mask(base, height, width)
        return resized.clamp(0.0, 1.0)


def _adjust_tile_controls(
    width: int,
    height: int,
    controls: TileSEGSControls,
) -> tuple[int, int]:
    """Clamp tile controls to valid values for the image dimensions."""

    bbox_size = min(controls.bbox_size, width, height)
    if bbox_size != controls.bbox_size:
        LOGGER.warning(
            "Tile & Tag SEGS clamped bbox_size to image dimensions",
            extra={
                "operation": "tile_and_tag_segs",
                "requested_bbox_size": controls.bbox_size,
                "bbox_size": bbox_size,
            },
        )
    min_overlap = controls.min_overlap
    if bbox_size <= 2 * min_overlap:
        adjusted = max(0, (bbox_size // 2) - 1)
        LOGGER.warning(
            "Tile & Tag SEGS adjusted min_overlap for valid tile stepping",
            extra={
                "operation": "tile_and_tag_segs",
                "requested_min_overlap": min_overlap,
                "min_overlap": adjusted,
                "bbox_size": bbox_size,
            },
        )
        min_overlap = adjusted
    return bbox_size, min_overlap


def _tile_regions(
    *,
    width: int,
    height: int,
    bbox_size: int,
    min_overlap: int,
    crop_factor: float,
) -> tuple[TileRegion, ...]:
    """Return row-major tile regions covering the image."""

    x_positions = _axis_positions(width, bbox_size, min_overlap)
    y_positions = _axis_positions(height, bbox_size, min_overlap)
    regions: list[TileRegion] = []
    for top in y_positions:
        for left in x_positions:
            bbox = BoundingBox(left, top, left + bbox_size, top + bbox_size)
            regions.append(
                TileRegion(
                    bbox=bbox,
                    crop_region=crop_region_for_bbox(
                        bbox,
                        image_height=height,
                        image_width=width,
                        crop_factor=crop_factor,
                    ),
                )
            )
    return tuple(regions)


def _axis_positions(length: int, bbox_size: int, min_overlap: int) -> tuple[int, ...]:
    """Return tile starts for one axis with even overlap distribution."""

    if bbox_size >= length:
        return (0,)
    step = bbox_size - min_overlap
    tile_count = max(1, math.ceil(length / step))
    overlap_sum = bbox_size * tile_count - length
    if overlap_sum < 0:
        tile_count += 1
        overlap_sum = bbox_size * tile_count - length
    overlap = 0 if tile_count == 1 else int(overlap_sum / (tile_count - 1))
    if overlap == bbox_size:
        return (0,)
    starts = []
    position = 0
    for _index in range(tile_count):
        starts.append(min(position, length - bbox_size))
        position += bbox_size - overlap
    return tuple(dict.fromkeys(starts))


def _tile_crop_mask(tile_region: TileRegion) -> torch.Tensor:
    """Return a rectangular tile mask in crop-region coordinates."""

    crop = tile_region.crop_region
    bbox = tile_region.bbox
    mask = torch.zeros((crop.height, crop.width), dtype=torch.float32)
    mask[
        bbox.top - crop.top : bbox.bottom - crop.top,
        bbox.left - crop.left : bbox.right - crop.left,
    ] = 1.0
    return mask
