# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for deterministic contextual diffusion planning."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.contextual_diffusion import (
    ContextualDiffusionControls,
    build_contextual_diffusion_plan,
    fit_context_shape,
)
from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment
from simple_syrup.domain.spatial_views import SpatialViewKind


def test_global_view_fits_maximum_dimension_without_boxing() -> None:
    """The whole canvas keeps its aspect ratio and uses no artificial padding."""

    assert fit_context_shape(384, 256, 128) == (128, 86)
    assert fit_context_shape(64, 96, 128) == (64, 96)


def test_plan_exposes_one_full_source_reduced_model_view() -> None:
    """Describe the Contextual global evaluation through canonical geometry."""

    plan = build_contextual_diffusion_plan(
        latent_width=96,
        latent_height=64,
        controls=_controls(),
        segs=None,
    )

    assert plan.global_view.kind is SpatialViewKind.CONTEXTUAL_GLOBAL
    assert (
        plan.global_view.source_x,
        plan.global_view.source_y,
        plan.global_view.source_width,
        plan.global_view.source_height,
    ) == (0, 0, 96, 64)
    assert (plan.global_view.model_width, plan.global_view.model_height) == (32, 22)


def test_connected_segs_replace_regular_grid_with_guided_tile_plan() -> None:
    """Contextual Diffusion delegates its tile path to SEGS-guided planning."""

    large = torch.ones((64, 64), dtype=torch.float32)
    small = torch.zeros((64, 64), dtype=torch.float32)
    small[24:40, 24:40] = 1.0
    plan = build_contextual_diffusion_plan(
        latent_width=64,
        latent_height=64,
        controls=_controls(),
        segs=((64, 64), (_segment(large, 0.8), _segment(small, 0.9))),
    )

    assert len(plan.tile_plan.tiles) > 1
    assert all(tile.weight_mask is not None for tile in plan.tile_plan.tiles)


def test_missing_segs_uses_regular_tiled_diffusion_plan() -> None:
    """The optional SEGS input preserves ordinary tiled diffusion as fallback."""

    plan = build_contextual_diffusion_plan(
        latent_width=64,
        latent_height=64,
        controls=_controls(),
        segs=None,
    )

    assert len(plan.tile_plan.tiles) > 1
    assert all(tile.weight_mask is None for tile in plan.tile_plan.tiles)


def test_invalid_overlap_fails_before_planning() -> None:
    """An overlap that cannot advance a context is rejected explicitly."""

    with pytest.raises(ValueError, match="latent_context_overlap"):
        build_contextual_diffusion_plan(
            latent_width=64,
            latent_height=64,
            controls=_controls(latent_context_overlap=32),
            segs=None,
        )


def _controls(
    *,
    latent_context_overlap: int = 8,
) -> ContextualDiffusionControls:
    """Return compact valid controls for planner tests."""

    return ContextualDiffusionControls(
        latent_context_size=32,
        latent_context_overlap=latent_context_overlap,
        latent_context_batch_size=2,
        global_weight=1.0,
        global_steps=1,
        global_decay=0.5,
    )


def _segment(mask: torch.Tensor, confidence: float) -> Segment:
    """Return a full-canvas SEG for a test mask."""

    height, width = mask.shape
    crop = CropRegion(0, 0, width, height)
    return Segment(
        cropped_image=None,
        cropped_mask=mask,
        confidence=confidence,
        crop_region=crop,
        bbox=BoundingBox(*crop),
        label="region",
    )
