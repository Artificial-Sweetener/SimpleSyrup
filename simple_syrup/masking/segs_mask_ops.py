# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Mask and crop helpers for SEGS detection and detailing."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..domain.segs import BoundingBox, CropRegion


def validate_single_image(image: object, operation: str) -> torch.Tensor:
    """Return a validated single-image BHWC tensor."""

    image_batch = validate_image_batch(image, operation)
    batch_size = int(image_batch.shape[0])
    if batch_size != 1:
        raise ValueError(
            f"{operation} currently supports one image at a time; "
            f"received batch size {batch_size}."
        )
    return image_batch


def validate_image_batch(image: object, operation: str) -> torch.Tensor:
    """Return a validated BHWC image batch tensor."""

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{operation} requires a torch IMAGE tensor.")
    if image.ndim != 4:
        raise ValueError(f"{operation} requires a BHWC IMAGE tensor.")
    batch_size = int(image.shape[0])
    if batch_size < 1:
        raise ValueError(f"{operation} requires at least one image.")
    channels = int(image.shape[-1])
    if channels < 1:
        raise ValueError(f"{operation} requires at least one image channel.")
    return image.float().clamp(0.0, 1.0)


def iter_single_images(image_batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return one-image BHWC slices from a validated image batch."""

    return tuple(
        image_batch[index : index + 1] for index in range(int(image_batch.shape[0]))
    )


def rectangular_mask(height: int, width: int, bbox: BoundingBox) -> torch.Tensor:
    """Create a full-image mask filled inside a bounding box."""

    mask = torch.zeros((height, width), dtype=torch.float32)
    mask[bbox.top : bbox.bottom, bbox.left : bbox.right] = 1.0
    return mask


def normalize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Normalize a mask to an unbatched HW tensor in float range."""

    working = mask.float()
    if working.ndim == 3:
        working = working[0]
    if working.ndim != 2:
        raise ValueError("Detection mask must be HW or BHW shaped.")
    if int(working.shape[0]) != height or int(working.shape[1]) != width:
        working = (
            F.interpolate(
                working.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
    return working.clamp(0.0, 1.0)


def dilate_mask(mask: torch.Tensor, dilation: int) -> torch.Tensor:
    """Morph an HW mask by a signed pixel radius."""

    if dilation == 0:
        return mask.float().clamp(0.0, 1.0)
    radius = abs(dilation)
    kernel_size = radius * 2 + 1
    padded = F.pad(
        mask.float().unsqueeze(0).unsqueeze(0),
        (radius, radius, radius, radius),
        value=0.0,
    )
    if dilation > 0:
        return (
            F.max_pool2d(padded, kernel_size=kernel_size, stride=1)
            .squeeze(0)
            .squeeze(0)
        )
    return (
        (-F.max_pool2d(-padded, kernel_size=kernel_size, stride=1))
        .squeeze(0)
        .squeeze(0)
    )


def feather_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Soften an HW mask edge by a pixel radius."""

    if radius < 0:
        raise ValueError("feather radius must be greater than or equal to 0.")
    if radius == 0:
        return mask.float().clamp(0.0, 1.0)
    kernel_size = radius * 2 + 1
    blurred = (
        F.avg_pool2d(
            mask.float().unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=radius,
            count_include_pad=False,
        )
        .squeeze(0)
        .squeeze(0)
    )
    return blurred.clamp(0.0, 1.0)


def crop_region_for_bbox(
    bbox: BoundingBox,
    image_height: int,
    image_width: int,
    crop_factor: float,
) -> CropRegion:
    """Expand a bbox around its center or select the full image for zero."""

    if crop_factor == 0.0:
        return CropRegion(0, 0, image_width, image_height)
    if crop_factor < 1.0:
        raise ValueError("crop_factor must be 0 or greater than or equal to 1.0.")
    center_x = (bbox.left + bbox.right) / 2.0
    center_y = (bbox.top + bbox.bottom) / 2.0
    crop_width = bbox.width * crop_factor
    crop_height = bbox.height * crop_factor
    left = max(0, int(round(center_x - crop_width / 2.0)))
    top = max(0, int(round(center_y - crop_height / 2.0)))
    right = min(image_width, int(round(center_x + crop_width / 2.0)))
    bottom = min(image_height, int(round(center_y + crop_height / 2.0)))
    if right <= left:
        right = min(image_width, left + 1)
    if bottom <= top:
        bottom = min(image_height, top + 1)
    return CropRegion(left, top, right, bottom)


def crop_image(image: torch.Tensor, region: CropRegion) -> torch.Tensor:
    """Crop a BHWC image tensor by a crop region."""

    return image[:, region.top : region.bottom, region.left : region.right, :]


def crop_mask(mask: torch.Tensor, region: CropRegion) -> torch.Tensor:
    """Crop an HW mask tensor by a crop region."""

    return mask[region.top : region.bottom, region.left : region.right]


def resize_image(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize a BHWC image tensor with bilinear interpolation."""

    resized = F.interpolate(
        image.movedim(-1, 1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.movedim(1, -1).clamp(0.0, 1.0)


def resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize an HW or BHW mask tensor with bilinear interpolation."""

    if mask.ndim == 2:
        working = mask.unsqueeze(0)
    elif mask.ndim == 3:
        working = mask
    else:
        raise ValueError("Mask must be HW or BHW shaped.")
    resized = F.interpolate(
        working.unsqueeze(1).float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    if mask.ndim == 2:
        return resized.squeeze(0).clamp(0.0, 1.0)
    return resized.clamp(0.0, 1.0)
