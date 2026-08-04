# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for projecting authoritative sampling contexts into SEGS."""

from __future__ import annotations

import torch

from simple_syrup.domain.context_segs import context_segs_from_tile_plan
from simple_syrup.domain.segs import BoundingBox, coerce_segs
from simple_syrup.domain.tiled_diffusion import build_tiled_diffusion_plan
from simple_syrup.services.simple_preview_segs_service import SimplePreviewSEGSService


def test_tile_plan_projects_exact_non_global_context_rectangles() -> None:
    """Every evaluated tile becomes one full rectangular SEG in image space."""

    plan = build_tiled_diffusion_plan(
        latent_width=64,
        latent_height=32,
        tile_width=32,
        tile_height=24,
        overlap=8,
        tile_batch_size=2,
    )

    header, contexts = context_segs_from_tile_plan(
        plan,
        image_height=256,
        image_width=512,
    )

    assert header == (256, 512)
    assert not contexts.is_materialized
    assert len(contexts) == len(plan.tiles)
    assert not contexts.is_materialized
    assert [context.label for context in contexts] == [
        f"context_{index:03d}" for index in range(1, len(contexts) + 1)
    ]
    assert all(context.crop_region != (0, 0, 512, 256) for context in contexts)
    for tile, context in zip(plan.tiles, contexts, strict=True):
        assert context.crop_region == (
            tile.x * 8,
            tile.y * 8,
            (tile.x + tile.width) * 8,
            (tile.y + tile.height) * 8,
        )
        assert context.bbox == BoundingBox(*context.crop_region)
        mask = context.cropped_mask
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.uint8
        assert tuple(mask.shape) == (
            context.crop_region.height,
            context.crop_region.width,
        )
        assert bool(mask.all())
    assert contexts.is_materialized


def test_lazy_contexts_feed_simple_preview_segs_directly() -> None:
    """Simple Preview SEGS materializes and displays a connected context output."""

    plan = build_tiled_diffusion_plan(
        latent_width=16,
        latent_height=8,
        tile_width=8,
        tile_height=8,
        overlap=2,
        tile_batch_size=2,
    )
    contexts_segs = context_segs_from_tile_plan(
        plan,
        image_height=64,
        image_width=128,
    )

    document = SimplePreviewSEGSService().build(
        image=torch.zeros((1, 64, 128, 3)),
        segs=coerce_segs(contexts_segs),
    )

    assert contexts_segs[1].is_materialized
    assert len(document.regions) == len(plan.tiles)
    assert all(region.label.startswith("context_") for region in document.regions)
