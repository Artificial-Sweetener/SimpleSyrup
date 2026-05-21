# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Image crop compositing helpers for detailer services."""

from __future__ import annotations

import torch

from ..domain.segs import CropRegion


def composite_crop(
    image: torch.Tensor,
    crop: torch.Tensor,
    mask: torch.Tensor,
    region: CropRegion,
) -> torch.Tensor:
    """Alpha-composite a detailed crop back into a BHWC image tensor."""

    if image.ndim != 4 or crop.ndim != 4:
        raise ValueError("image and crop must be BHWC tensors.")
    if mask.ndim != 2:
        raise ValueError("composite mask must be an HW tensor.")
    expected_height = region.height
    expected_width = region.width
    if int(crop.shape[1]) != expected_height or int(crop.shape[2]) != expected_width:
        raise ValueError("detailed crop dimensions must match crop region.")
    if int(mask.shape[0]) != expected_height or int(mask.shape[1]) != expected_width:
        raise ValueError("composite mask dimensions must match crop region.")

    output = image.clone()
    alpha = mask.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
    alpha_bhwc = alpha.unsqueeze(0).unsqueeze(-1)
    original = output[:, region.top : region.bottom, region.left : region.right, :]
    blended = crop.to(device=image.device, dtype=image.dtype) * alpha_bhwc
    blended = blended + original * (1.0 - alpha_bhwc)
    output[:, region.top : region.bottom, region.left : region.right, :] = blended
    return output.clamp(0.0, 1.0)
