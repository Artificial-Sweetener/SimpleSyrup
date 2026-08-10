# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Constrain semantic tiled diffusion by authored regional composition masks."""

from __future__ import annotations

import torch

from .segs import coerce_segs
from .segs_tiled_diffusion import segs_ownership_masks, validate_segs_aspect_ratio
from .semantic_tiled_diffusion import build_semantic_tiled_diffusion_plan
from .tiled_diffusion import TiledDiffusionPlan

REGIONAL_PLANNING_THRESHOLD = 0.5


def build_region_constrained_tiled_diffusion_plan(
    *,
    region_masks: torch.Tensor,
    segs: object | None,
    latent_width: int,
    latent_height: int,
    tile_width: int,
    tile_height: int,
    overlap: int,
    tile_batch_size: int,
) -> TiledDiffusionPlan:
    """Build tiles split wherever regional composition or optional SEGS change."""

    region_ownership = regional_composition_ownership_masks(
        region_masks,
        latent_height=latent_height,
        latent_width=latent_width,
    )
    ownership_masks = region_ownership
    if segs is not None:
        native_segs = coerce_segs(segs)
        validate_segs_aspect_ratio(native_segs, latent_height, latent_width)
        semantic_ownership = segs_ownership_masks(
            native_segs,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        ownership_masks = _intersect_partitions(
            region_ownership,
            semantic_ownership,
        )
    return build_semantic_tiled_diffusion_plan(
        ownership_masks=ownership_masks,
        latent_width=latent_width,
        latent_height=latent_height,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
        tile_batch_size=tile_batch_size,
        merge_across_masks=False,
    )


def regional_composition_ownership_masks(
    region_masks: torch.Tensor,
    *,
    latent_height: int,
    latent_width: int,
) -> tuple[torch.Tensor, ...]:
    """Partition the canvas by every distinct active regional-mask combination."""

    if region_masks.ndim != 3:
        raise ValueError("Regional tile planning requires a BHW mask batch.")
    if tuple(region_masks.shape[1:]) != (latent_height, latent_width):
        raise ValueError(
            "Regional tile planning masks must match latent shape "
            f"{latent_height}x{latent_width}."
        )
    membership = (region_masks.detach().cpu() >= REGIONAL_PLANNING_THRESHOLD).permute(
        1, 2, 0
    )
    flattened = membership.reshape(latent_height * latent_width, -1)
    signatures, inverse = torch.unique(
        flattened,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    del signatures
    labels = inverse.reshape(latent_height, latent_width)
    return tuple(labels == index for index in range(int(labels.max().item()) + 1))


def _intersect_partitions(
    first: tuple[torch.Tensor, ...],
    second: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Return non-empty intersections of two complete ownership partitions."""

    intersections = tuple(
        intersection
        for first_mask in first
        for second_mask in second
        if bool((intersection := torch.logical_and(first_mask, second_mask)).any())
    )
    if not intersections:
        raise ValueError("Regional and SEGS ownership produced no tile coverage.")
    return intersections
