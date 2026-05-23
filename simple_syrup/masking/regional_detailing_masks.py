# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file incorporate behavior derived from
# multidiffusion-upscaler-for-automatic1111. See third_party/manifest.toml and
# third_party/NOTICE.md.

"""Mask conversion helpers for regional SEGS detailing."""

from __future__ import annotations

import math

import torch

from ..domain.regional_detailing import (
    ImageRegion,
    LatentBox,
    LatentRegion,
    SegmentConditioningPair,
)
from ..domain.segs import CropRegion
from .detailer_masks import gaussian_feather_mask
from .segs_mask_ops import feather_mask, resize_mask

OPERATION = "Detail SEGS as Regions"


def build_image_regions(
    pairs: tuple[SegmentConditioningPair, ...],
    *,
    image_height: int,
    image_width: int,
) -> tuple[ImageRegion, ...]:
    """Create full-image masks from paired SEGS crop-local masks."""

    return tuple(
        _image_region_from_pair(
            pair,
            image_height=image_height,
            image_width=image_width,
        )
        for pair in pairs
    )


def build_latent_regions(
    image_regions: tuple[ImageRegion, ...],
    *,
    latent_height: int,
    latent_width: int,
    device: torch.device,
    dtype: torch.dtype,
    latent_feather: int,
) -> tuple[LatentRegion, ...]:
    """Convert image-space regions into latent-space masks and boxes."""

    if latent_height < 1 or latent_width < 1:
        raise ValueError("latent dimensions must be positive.")
    latent_regions: list[LatentRegion] = []
    for region in image_regions:
        latent_mask = resize_mask(region.image_mask, latent_height, latent_width)
        if latent_feather > 0:
            latent_mask = feather_mask(latent_mask, latent_feather)
        latent_mask = latent_mask.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        latent_box = latent_box_from_mask(
            latent_mask,
            region_index=region.index,
            label=region.label,
        )
        latent_regions.append(
            LatentRegion(
                index=region.index,
                label=region.label,
                latent_box=latent_box,
                latent_mask=latent_mask,
                positive=region.positive,
            )
        )
    return tuple(latent_regions)


def scale_image_regions(
    image_regions: tuple[ImageRegion, ...],
    *,
    image_height: int,
    image_width: int,
) -> tuple[ImageRegion, ...]:
    """Resize full-image region masks into a scaled image coordinate space."""

    if image_height < 1 or image_width < 1:
        raise ValueError("scaled image dimensions must be positive.")
    return tuple(
        ImageRegion(
            index=region.index,
            label=region.label,
            crop_region=_scale_crop_region(
                region.crop_region,
                source_height=int(region.image_mask.shape[0]),
                source_width=int(region.image_mask.shape[1]),
                target_height=image_height,
                target_width=image_width,
            ),
            image_mask=resize_mask(region.image_mask, image_height, image_width),
            positive=region.positive,
        )
        for region in image_regions
    )


def latent_box_from_mask(
    mask: torch.Tensor,
    *,
    region_index: int,
    label: str,
) -> LatentBox:
    """Return the tight latent box around a non-empty HW mask."""

    if mask.ndim != 2:
        raise ValueError("latent region mask must be HW shaped.")
    coordinates = torch.nonzero(mask > 0, as_tuple=False)
    if coordinates.numel() == 0:
        raise ValueError(
            f"{OPERATION} SEG {region_index} ('{label}') produced an empty "
            "latent region."
        )
    top = int(coordinates[:, 0].min().item())
    bottom = int(coordinates[:, 0].max().item()) + 1
    left = int(coordinates[:, 1].min().item())
    right = int(coordinates[:, 1].max().item()) + 1
    return LatentBox(
        x=left,
        y=top,
        width=right - left,
        height=bottom - top,
    )


def union_masks(masks: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Return the clamped union of one or more same-shaped HW masks."""

    if not masks:
        raise ValueError("at least one mask is required.")
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise ValueError("all masks must have the same shape.")
    union = torch.zeros_like(masks[0], dtype=torch.float32)
    for mask in masks:
        union = torch.maximum(union, mask.float())
    return union.clamp(0.0, 1.0)


def feather_image_mask(mask: torch.Tensor, feather: int) -> torch.Tensor:
    """Feather an image-space mask while preserving the HW contract."""

    return gaussian_feather_mask(mask, feather)


def proportional_latent_box(
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
    image_height: int,
    image_width: int,
    latent_height: int,
    latent_width: int,
) -> LatentBox:
    """Convert an image-space box to a clamped latent-space box."""

    if image_height < 1 or image_width < 1:
        raise ValueError("image dimensions must be positive.")
    x = max(0, min(latent_width - 1, math.floor(left * latent_width / image_width)))
    y = max(0, min(latent_height - 1, math.floor(top * latent_height / image_height)))
    x2 = max(x + 1, min(latent_width, math.ceil(right * latent_width / image_width)))
    y2 = max(
        y + 1, min(latent_height, math.ceil(bottom * latent_height / image_height))
    )
    return LatentBox(x=x, y=y, width=x2 - x, height=y2 - y)


def _scale_crop_region(
    region: CropRegion,
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> CropRegion:
    """Scale a crop region proportionally into a target image shape."""

    if source_height < 1 or source_width < 1:
        raise ValueError("source image dimensions must be positive.")
    left = max(
        0, min(target_width - 1, math.floor(region.left * target_width / source_width))
    )
    top = max(
        0,
        min(target_height - 1, math.floor(region.top * target_height / source_height)),
    )
    right = max(
        left + 1,
        min(target_width, math.ceil(region.right * target_width / source_width)),
    )
    bottom = max(
        top + 1,
        min(target_height, math.ceil(region.bottom * target_height / source_height)),
    )
    return CropRegion(left, top, right, bottom)


def _image_region_from_pair(
    pair: SegmentConditioningPair,
    *,
    image_height: int,
    image_width: int,
) -> ImageRegion:
    """Paste one pair's crop-local mask into full-image mask space."""

    crop_mask = _crop_mask_to_hw(
        pair.segment.cropped_mask,
        height=pair.segment.crop_region.height,
        width=pair.segment.crop_region.width,
        index=pair.index,
        label=pair.segment.label,
    )
    full_mask = torch.zeros((image_height, image_width), dtype=torch.float32)
    region = pair.segment.crop_region
    full_mask[region.top : region.bottom, region.left : region.right] = crop_mask
    return ImageRegion(
        index=pair.index,
        label=pair.segment.label,
        crop_region=region,
        image_mask=full_mask.clamp(0.0, 1.0),
        positive=pair.positive,
    )


def _crop_mask_to_hw(
    mask: object,
    *,
    height: int,
    width: int,
    index: int,
    label: str,
) -> torch.Tensor:
    """Return a crop-local HW mask resized to the segment crop region."""

    mask_tensor = torch.as_tensor(mask).float()
    if mask_tensor.ndim == 3 and int(mask_tensor.shape[0]) == 1:
        mask_tensor = mask_tensor[0]
    if mask_tensor.ndim != 2:
        raise ValueError(
            f"{OPERATION} SEG {index} ('{label}') cropped_mask must be HW or "
            "single-batch BHW shaped."
        )
    resized = resize_mask(mask_tensor, height, width)
    if not torch.any(resized > 0):
        raise ValueError(
            f"{OPERATION} SEG {index} ('{label}') produced an empty image region."
        )
    return resized.float().clamp(0.0, 1.0)
