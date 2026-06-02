# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file incorporate behavior derived from
# multidiffusion-upscaler-for-automatic1111. See third_party/manifest.toml and
# third_party/NOTICE.md.

"""Pure tiled diffusion planning and mode validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

TILED_DIFFUSION_MODES = ("multidiffusion", "mixture_of_diffusers")


@dataclass(frozen=True)
class LatentTile:
    """Describe one rectangular latent-space tile."""

    x: int
    y: int
    width: int
    height: int

    @property
    def slicer(self) -> tuple[slice, slice, slice, slice]:
        """Return a tensor slicer for this tile on BCHW latents."""

        return (
            slice(None),
            slice(None),
            slice(self.y, self.y + self.height),
            slice(self.x, self.x + self.width),
        )


@dataclass(frozen=True)
class TiledDiffusionPlan:
    """Describe deterministic latent tiles and balanced tile batches."""

    latent_width: int
    latent_height: int
    tile_width: int
    tile_height: int
    overlap: int
    requested_tile_batch_size: int
    tile_batch_size: int
    tiles: tuple[LatentTile, ...]
    batches: tuple[tuple[LatentTile, ...], ...]


def build_tiled_diffusion_plan(
    latent_width: int,
    latent_height: int,
    tile_width: int,
    tile_height: int,
    overlap: int,
    tile_batch_size: int,
) -> TiledDiffusionPlan:
    """Build a deterministic latent tile plan for tiled denoising."""

    _validate_plan_inputs(
        latent_width=latent_width,
        latent_height=latent_height,
        tile_width=tile_width,
        tile_height=tile_height,
        tile_batch_size=tile_batch_size,
    )
    effective_tile_width = min(tile_width, latent_width)
    effective_tile_height = min(tile_height, latent_height)
    max_effective_overlap = min(effective_tile_width, effective_tile_height) - 4
    effective_overlap = max(0, min(overlap, max_effective_overlap))

    tiles = _split_tiles(
        latent_width=latent_width,
        latent_height=latent_height,
        tile_width=effective_tile_width,
        tile_height=effective_tile_height,
        overlap=effective_overlap,
    )
    batches, effective_tile_batch_size = _batch_tiles(tiles, tile_batch_size)
    return TiledDiffusionPlan(
        latent_width=latent_width,
        latent_height=latent_height,
        tile_width=effective_tile_width,
        tile_height=effective_tile_height,
        overlap=effective_overlap,
        requested_tile_batch_size=tile_batch_size,
        tile_batch_size=effective_tile_batch_size,
        tiles=tiles,
        batches=batches,
    )


def tile_is_splittable(
    latent_width: int,
    latent_height: int,
    tile_width: int,
    tile_height: int,
    overlap: int,
) -> bool:
    """Return whether the tile grid produces more than one tile."""

    plan = build_tiled_diffusion_plan(
        latent_width=latent_width,
        latent_height=latent_height,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
        tile_batch_size=1,
    )
    return len(plan.tiles) > 1


def validate_tiled_diffusion_mode(diffusion_mode: str) -> None:
    """Reject unsupported tiled diffusion modes."""

    if diffusion_mode in TILED_DIFFUSION_MODES:
        return
    supported = ", ".join(TILED_DIFFUSION_MODES)
    raise ValueError(
        f"diffusion_mode must be one of: {supported}; got {diffusion_mode!r}."
    )


def gaussian_tile_weights(
    tile_width: int,
    tile_height: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return Mixture of Diffusers Gaussian tile weights."""

    if tile_width < 4:
        raise ValueError("tile_width must be at least 4.")
    if tile_height < 4:
        raise ValueError("tile_height must be at least 4.")

    x_values = torch.arange(tile_width, device=device, dtype=torch.float64)
    y_values = torch.arange(tile_height, device=device, dtype=torch.float64)
    variance = 0.01

    x_midpoint = (tile_width - 1) / 2
    y_midpoint = tile_height / 2
    denominator = math.sqrt(2 * math.pi * variance)
    x_probs = (
        torch.exp(
            -((x_values - x_midpoint) * (x_values - x_midpoint))
            / (tile_width * tile_width)
            / (2 * variance)
        )
        / denominator
    )
    y_probs = (
        torch.exp(
            -((y_values - y_midpoint) * (y_values - y_midpoint))
            / (tile_width * tile_width)
            / (2 * variance)
        )
        / denominator
    )
    return torch.outer(y_probs, x_probs).to(dtype=dtype)


def _validate_plan_inputs(
    *,
    latent_width: int,
    latent_height: int,
    tile_width: int,
    tile_height: int,
    tile_batch_size: int,
) -> None:
    """Reject invalid tile planning values before sampling."""

    if latent_width < 1:
        raise ValueError("latent_width must be at least 1.")
    if latent_height < 1:
        raise ValueError("latent_height must be at least 1.")
    if tile_width < 4:
        raise ValueError("tile_width must be at least 4.")
    if tile_height < 4:
        raise ValueError("tile_height must be at least 4.")
    if tile_batch_size < 1:
        raise ValueError("tile_batch_size must be at least 1.")


def _split_tiles(
    *,
    latent_width: int,
    latent_height: int,
    tile_width: int,
    tile_height: int,
    overlap: int,
) -> tuple[LatentTile, ...]:
    """Split a latent with a symmetric row-major grid formula."""

    cols = math.ceil((latent_width - overlap) / (tile_width - overlap))
    rows = math.ceil((latent_height - overlap) / (tile_height - overlap))
    dx = (latent_width - tile_width) / (cols - 1) if cols > 1 else 0
    dy = (latent_height - tile_height) / (rows - 1) if rows > 1 else 0

    tiles: list[LatentTile] = []
    for row in range(rows):
        y = min(int(row * dy), latent_height - tile_height)
        for col in range(cols):
            x = min(int(col * dx), latent_width - tile_width)
            tiles.append(LatentTile(x, y, tile_width, tile_height))
    return tuple(tiles)


def _batch_tiles(
    tiles: tuple[LatentTile, ...],
    requested_tile_batch_size: int,
) -> tuple[tuple[tuple[LatentTile, ...], ...], int]:
    """Group tiles using balanced effective tile batch sizing."""

    num_batches = math.ceil(len(tiles) / requested_tile_batch_size)
    effective_tile_batch_size = math.ceil(len(tiles) / num_batches)
    batches = tuple(
        tiles[
            index * effective_tile_batch_size : (index + 1) * effective_tile_batch_size
        ]
        for index in range(num_batches)
    )
    return batches, effective_tile_batch_size
