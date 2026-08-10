# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for region-constrained semantic tile planning."""

from __future__ import annotations

import torch

from simple_syrup.domain.regional_tiled_diffusion import (
    build_region_constrained_tiled_diffusion_plan,
    regional_composition_ownership_masks,
)
from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment


def test_composition_partitions_preserve_overlapping_mask_membership() -> None:
    """Every distinct global/regional conditioning combination owns one zone."""

    masks = torch.zeros((2, 4, 8))
    masks[0, :, :6] = 1.0
    masks[1, :, 2:] = 1.0

    ownership = regional_composition_ownership_masks(
        masks,
        latent_height=4,
        latent_width=8,
    )

    assert len(ownership) == 3
    coverage = torch.stack([mask.to(dtype=torch.int32) for mask in ownership]).sum(
        dim=0
    )
    assert torch.equal(coverage, torch.ones((4, 8), dtype=torch.int32))


def test_region_plan_never_merges_across_composition_boundaries() -> None:
    """Broad authored regions remain distinct even when one window could hold both."""

    masks = torch.zeros((2, 4, 8))
    masks[0, :, :4] = 1.0
    masks[1, :, 4:] = 1.0

    plan = build_region_constrained_tiled_diffusion_plan(
        region_masks=masks,
        segs=None,
        latent_width=8,
        latent_height=4,
        tile_width=8,
        tile_height=4,
        overlap=0,
        tile_batch_size=4,
    )

    assert len(plan.tiles) == 2
    coverage = torch.zeros((4, 8), dtype=torch.float32)
    for tile in plan.tiles:
        assert tile.weight_mask is not None
        coverage[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] += (
            tile.weight_mask
        )
    assert torch.equal(coverage, torch.ones((4, 8)))


def test_segs_subdivide_each_regional_composition_zone() -> None:
    """SEGS geometry is intersected with region ownership instead of replacing it."""

    masks = torch.zeros((2, 8, 8))
    masks[0, :, :4] = 1.0
    masks[1, :, 4:] = 1.0
    seg_mask = torch.zeros((8, 8))
    seg_mask[:4, :] = 1.0
    segs = ((8, 8), (_segment(seg_mask),))

    plan = build_region_constrained_tiled_diffusion_plan(
        region_masks=masks,
        segs=segs,
        latent_width=8,
        latent_height=8,
        tile_width=8,
        tile_height=8,
        overlap=0,
        tile_batch_size=4,
    )

    assert len(plan.tiles) == 4
    assert all(tile.weight_mask is not None for tile in plan.tiles)


def _segment(mask: torch.Tensor) -> Segment:
    """Return one full-canvas SEG for intersection tests."""

    height, width = mask.shape
    crop = CropRegion(0, 0, width, height)
    return Segment(
        cropped_image=None,
        cropped_mask=mask,
        confidence=1.0,
        crop_region=crop,
        bbox=BoundingBox(*crop),
        label="top",
    )
