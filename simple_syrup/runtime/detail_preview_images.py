# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Convert and align image assets used by detail sampling previews."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..domain.segs import CropRegion

CropBox = tuple[int, int, int, int]


def image_tensor_to_rgb_pil(image: torch.Tensor) -> Image.Image:
    """Convert a single-image BHWC tensor to an RGB PIL image."""

    if image.ndim != 4:
        raise ValueError("detail preview image must be a BHWC tensor.")
    if int(image.shape[0]) != 1:
        raise ValueError("detail preview image must contain exactly one image.")
    if int(image.shape[-1]) < 1:
        raise ValueError("detail preview image must contain at least one channel.")
    array = image[0].detach().cpu().float().clamp(0.0, 1.0).numpy()
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] >= 3:
        array = array[..., :3]
    else:
        array = np.repeat(array[..., :1], 3, axis=-1)
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


def normalize_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    """Normalize an HW or single-item BHW mask tensor to HW float."""

    working = mask.detach().float()
    if working.ndim == 3 and int(working.shape[0]) == 1:
        working = working[0]
    if working.ndim != 2:
        raise ValueError("detail preview work mask must be an HW tensor.")
    return working


def detail_alpha_mask(
    mask: torch.Tensor,
    *,
    source_size: tuple[int, int],
    preview_size: tuple[int, int],
    sampled_box: CropBox,
    target_size: tuple[int, int],
) -> Image.Image:
    """Return an alpha mask aligned to the sampled preview paste box."""

    working = normalize_mask_tensor(mask).detach().cpu().clamp(0.0, 1.0)
    mask_image = Image.fromarray((working.numpy() * 255.0).round().astype(np.uint8))
    if mask_image.size == source_size:
        preview_mask = mask_image.resize(preview_size, Image.Resampling.BILINEAR)
        return preview_mask.crop(sampled_box).resize(
            target_size,
            Image.Resampling.BILINEAR,
        )
    return mask_image.resize(target_size, Image.Resampling.BILINEAR)


def validate_crop_region(
    crop_region: CropRegion,
    source_width: int,
    source_height: int,
) -> None:
    """Reject crop regions that cannot be mapped into the source image."""

    if crop_region.left < 0 or crop_region.top < 0:
        raise ValueError("crop_region left and top must be non-negative.")
    if crop_region.right <= crop_region.left or crop_region.bottom <= crop_region.top:
        raise ValueError("crop_region right/bottom must be greater than left/top.")
    if crop_region.right > source_width or crop_region.bottom > source_height:
        raise ValueError("crop_region must fit within the source image.")
