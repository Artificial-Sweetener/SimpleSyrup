# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file incorporate behavior derived from
# multidiffusion-upscaler-for-automatic1111. See third_party/manifest.toml and
# third_party/NOTICE.md.

"""Crop and resize tensor data only along its final spatial axes."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from ..domain.spatial_views import SpatialView
from ..domain.tiled_diffusion import LatentTile


def spatial_tile_slicer(tile: LatentTile, tensor_ndim: int) -> tuple[slice, ...]:
    """Return a slicer that crops only a tensor's final height and width axes."""

    return (
        (slice(None),) * (tensor_ndim - 2)
        + (slice(tile.y, tile.y + tile.height),)
        + (slice(tile.x, tile.x + tile.width),)
    )


def spatial_view_slicer(
    view: SpatialView,
    tensor_ndim: int,
) -> tuple[slice, ...]:
    """Return a slicer for one arbitrary spatial view source rectangle."""

    return (
        (slice(None),) * (tensor_ndim - 2)
        + (slice(view.source_y, view.source_bottom),)
        + (slice(view.source_x, view.source_right),)
    )


def resize_spatial_tensor(
    tensor: torch.Tensor,
    *,
    height: int,
    width: int,
    mode: str,
) -> torch.Tensor:
    """Resize only the final two axes of a 4D or singleton-depth 5D tensor."""

    if tensor.shape[-2:] == (height, width):
        return tensor
    leading_shape = tensor.shape[:-2]
    flattened = tensor.reshape(-1, 1, tensor.shape[-2], tensor.shape[-1])
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    resized = functional.interpolate(
        flattened,
        size=(height, width),
        mode=mode,
        align_corners=align_corners,
    )
    return resized.reshape(*leading_shape, height, width)
