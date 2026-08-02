# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for SEGS-guided irregular tiled diffusion planning."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment
from simple_syrup.domain.segs_tiled_diffusion import (
    build_segs_guided_tiled_diffusion_plan,
    segment_mask_to_latent,
    segment_weight_to_latent,
)


def test_guided_plan_splits_oversized_region_with_bounded_rectangular_windows() -> None:
    """A large region becomes multiple bounded cores instead of one giant tile."""

    plan = build_segs_guided_tiled_diffusion_plan(
        segs=_full_mask_segs(64, 64),
        latent_width=64,
        latent_height=64,
        tile_width=32,
        tile_height=32,
        overlap=8,
        tile_batch_size=3,
    )

    assert len(plan.tiles) > 1
    assert all(tile.width == 32 and tile.height == 32 for tile in plan.tiles)
    assert all(tile.weight_mask is not None for tile in plan.tiles)
    coverage = torch.zeros((64, 64), dtype=torch.float32)
    for tile in plan.tiles:
        assert tile.weight_mask is not None
        coverage[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] += (
            tile.weight_mask
        )
    assert bool((coverage > 0).all())


def test_guided_plan_feathers_the_configured_overlap_between_irregular_cores() -> None:
    """Non-zero overlap produces fractional shared ownership weights."""

    plan = build_segs_guided_tiled_diffusion_plan(
        segs=_half_mask_segs(64, 64),
        latent_width=64,
        latent_height=64,
        tile_width=32,
        tile_height=32,
        overlap=8,
        tile_batch_size=4,
    )

    weights = [tile.weight_mask for tile in plan.tiles if tile.weight_mask is not None]
    assert any(bool(torch.any((weight > 0) & (weight < 1))) for weight in weights)


def test_guided_plan_prefers_smaller_overlapping_segs_for_ownership() -> None:
    """A nested small SEG remains an ownership core instead of being swallowed."""

    full = torch.ones((32, 32), dtype=torch.float32)
    small = torch.zeros((32, 32), dtype=torch.float32)
    small[8:24, 8:24] = 1.0
    segs = (
        (32, 32),
        (
            _segment(full, "large", 0.6),
            _segment(small, "small", 0.9),
        ),
    )

    plan = build_segs_guided_tiled_diffusion_plan(
        segs=segs,
        latent_width=32,
        latent_height=32,
        tile_width=32,
        tile_height=32,
        overlap=0,
        tile_batch_size=4,
    )

    assert len(plan.tiles) == 2
    assert all(tile.weight_mask is not None for tile in plan.tiles)


def test_guided_plan_merges_small_cores_without_comparing_tensor_values() -> None:
    """Small nearby cores merge through their indexes instead of tensor equality."""

    first = torch.zeros((32, 32), dtype=torch.float32)
    second = torch.zeros((32, 32), dtype=torch.float32)
    first[4:8, 4:8] = 1.0
    second[8:12, 8:12] = 1.0
    segs = (
        (32, 32),
        (
            _segment(first, "first", 1.0),
            _segment(second, "second", 1.0),
        ),
    )

    plan = build_segs_guided_tiled_diffusion_plan(
        segs=segs,
        latent_width=32,
        latent_height=32,
        tile_width=32,
        tile_height=32,
        overlap=0,
        tile_batch_size=4,
    )

    assert len(plan.tiles) == 1


def test_crop_local_mask_projects_directly_to_latent_space() -> None:
    """A small crop maps without materializing a full source-resolution mask."""

    crop = CropRegion(512, 256, 768, 512)
    segment = Segment(
        cropped_image=None,
        cropped_mask=torch.ones((256, 256), dtype=torch.float32),
        confidence=1.0,
        crop_region=crop,
        bbox=BoundingBox(*crop),
        label="small_region",
    )

    latent_mask = segment_mask_to_latent(
        segment,
        source_height=4096,
        source_width=4096,
        latent_width=512,
        latent_height=512,
    )

    assert int(latent_mask.sum().item()) == 32 * 32
    assert bool(latent_mask[32:64, 64:96].all())
    assert not bool(latent_mask[:32].any())
    assert not bool(latent_mask[:, :64].any())


def test_crop_local_weight_projection_preserves_soft_mask_values() -> None:
    """Semantic consumers can retain fractional SAM write ownership."""

    crop = CropRegion(0, 0, 8, 8)
    segment = Segment(
        cropped_image=None,
        cropped_mask=torch.full((8, 8), 0.25, dtype=torch.float32),
        confidence=1.0,
        crop_region=crop,
        bbox=BoundingBox(*crop),
        label="soft_region",
    )

    latent_weight = segment_weight_to_latent(
        segment,
        source_height=8,
        source_width=8,
        latent_width=8,
        latent_height=8,
    )

    assert torch.allclose(latent_weight, torch.full((8, 8), 0.25))


def test_guided_plan_rejects_mismatched_image_aspect_ratio() -> None:
    """SEGS from a different image fail before tiled sampling begins."""

    with pytest.raises(ValueError, match="aspect ratio"):
        build_segs_guided_tiled_diffusion_plan(
            segs=_full_mask_segs(16, 32),
            latent_width=32,
            latent_height=32,
            tile_width=16,
            tile_height=16,
            overlap=4,
            tile_batch_size=2,
        )


def _full_mask_segs(
    height: int, width: int
) -> tuple[tuple[int, int], tuple[Segment, ...]]:
    """Return one full-image SEG payload."""

    return ((height, width), (_segment(torch.ones((height, width)), "region", 1.0),))


def _half_mask_segs(
    height: int, width: int
) -> tuple[tuple[int, int], tuple[Segment, ...]]:
    """Return one left-half SEG payload with implicit background ownership."""

    mask = torch.zeros((height, width), dtype=torch.float32)
    mask[:, : width // 2] = 1.0
    return ((height, width), (_segment(mask, "left", 1.0),))


def _segment(mask: torch.Tensor, label: str, confidence: float) -> Segment:
    """Build one full-image Segment whose crop equals the source dimensions."""

    height, width = mask.shape
    region = CropRegion(0, 0, width, height)
    return Segment(
        cropped_image=None,
        cropped_mask=mask,
        confidence=confidence,
        crop_region=region,
        bbox=BoundingBox(0, 0, width, height),
        label=label,
    )
