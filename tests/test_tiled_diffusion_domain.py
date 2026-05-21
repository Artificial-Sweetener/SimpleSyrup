# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Mixture of Diffusers tiled diffusion domain behavior."""

from __future__ import annotations

import math

import pytest
import torch

from simple_syrup.domain.tiled_diffusion import (
    TILED_DIFFUSION_MODES,
    LatentTile,
    build_tiled_diffusion_plan,
    gaussian_tile_weights,
    tile_is_splittable,
    validate_tiled_diffusion_mode,
)


def test_tiled_diffusion_modes_match_node_contract() -> None:
    """Supported tiled diffusion modes are stable workflow-facing values."""

    assert TILED_DIFFUSION_MODES == ("multidiffusion", "mixture_of_diffusers")


@pytest.mark.parametrize("diffusion_mode", TILED_DIFFUSION_MODES)
def test_validate_tiled_diffusion_mode_accepts_supported_modes(
    diffusion_mode: str,
) -> None:
    """Supported tiled diffusion modes pass validation."""

    validate_tiled_diffusion_mode(diffusion_mode)


def test_validate_tiled_diffusion_mode_rejects_unsupported_mode() -> None:
    """Unsupported modes fail with an actionable field-specific error."""

    with pytest.raises(ValueError) as exc_info:
        validate_tiled_diffusion_mode("full_latent")

    message = str(exc_info.value)
    assert "diffusion_mode" in message
    assert "multidiffusion" in message
    assert "mixture_of_diffusers" in message
    assert "full_latent" in message


def test_plan_clamps_tile_size_to_latent_dimensions() -> None:
    """Requested tiles larger than the latent are clamped to latent dimensions."""

    plan = build_tiled_diffusion_plan(
        latent_width=12,
        latent_height=8,
        tile_width=96,
        tile_height=96,
        overlap=48,
        tile_batch_size=4,
    )

    assert plan.tile_width == 12
    assert plan.tile_height == 8
    assert plan.overlap == 48
    assert plan.tiles == (LatentTile(0, 0, 12, 8),)
    assert not tile_is_splittable(12, 8, 96, 96, 48)


def test_plan_clamps_overlap_against_requested_tile_dimensions() -> None:
    """Overlap clamps against requested tile dimensions."""

    plan = build_tiled_diffusion_plan(
        latent_width=8,
        latent_height=8,
        tile_width=96,
        tile_height=96,
        overlap=200,
        tile_batch_size=4,
    )

    assert plan.tile_width == 8
    assert plan.tile_height == 8
    assert plan.overlap == 92


def test_plan_generates_row_major_symmetric_tiles() -> None:
    """Tile positions follow deterministic row-major symmetric grid math."""

    plan = build_tiled_diffusion_plan(
        latent_width=20,
        latent_height=12,
        tile_width=8,
        tile_height=6,
        overlap=2,
        tile_batch_size=3,
    )

    assert plan.tiles == (
        LatentTile(0, 0, 8, 6),
        LatentTile(6, 0, 8, 6),
        LatentTile(12, 0, 8, 6),
        LatentTile(0, 3, 8, 6),
        LatentTile(6, 3, 8, 6),
        LatentTile(12, 3, 8, 6),
        LatentTile(0, 6, 8, 6),
        LatentTile(6, 6, 8, 6),
        LatentTile(12, 6, 8, 6),
    )
    assert tile_is_splittable(20, 12, 8, 6, 2)


def test_plan_covers_non_divisible_latent_edges() -> None:
    """Edge tiles clamp to the latent boundary and cover every pixel."""

    plan = build_tiled_diffusion_plan(
        latent_width=17,
        latent_height=11,
        tile_width=7,
        tile_height=5,
        overlap=1,
        tile_batch_size=2,
    )
    coverage = torch.zeros((plan.latent_height, plan.latent_width), dtype=torch.bool)

    for tile in plan.tiles:
        coverage[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] = True
        assert tile.x + tile.width <= plan.latent_width
        assert tile.y + tile.height <= plan.latent_height

    assert coverage.all()


def test_plan_uses_balanced_effective_batch_sizing() -> None:
    """Tile batches are balanced with an effective tile batch size."""

    plan = build_tiled_diffusion_plan(
        latent_width=20,
        latent_height=12,
        tile_width=8,
        tile_height=6,
        overlap=2,
        tile_batch_size=4,
    )

    assert plan.tile_batch_size == 3
    assert tuple(len(batch) for batch in plan.batches) == (3, 3, 3)


def test_gaussian_tile_weights_match_reference_formula() -> None:
    """Gaussian tile weights preserve the reference Mixture formula."""

    weights = gaussian_tile_weights(
        4,
        4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected = _reference_gaussian_weights(4, 4)

    assert weights.shape == (4, 4)
    assert torch.all(weights > 0)
    assert torch.allclose(weights, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("latent_width", "latent_height", "tile_width", "tile_height", "tile_batch_size"),
    [
        (0, 8, 4, 4, 1),
        (8, 0, 4, 4, 1),
        (8, 8, 3, 4, 1),
        (8, 8, 4, 3, 1),
        (8, 8, 4, 4, 0),
    ],
)
def test_plan_rejects_invalid_dimensions(
    latent_width: int,
    latent_height: int,
    tile_width: int,
    tile_height: int,
    tile_batch_size: int,
) -> None:
    """Invalid tile controls fail before sampling side effects."""

    with pytest.raises(ValueError):
        build_tiled_diffusion_plan(
            latent_width=latent_width,
            latent_height=latent_height,
            tile_width=tile_width,
            tile_height=tile_height,
            overlap=0,
            tile_batch_size=tile_batch_size,
        )


def _reference_gaussian_weights(tile_width: int, tile_height: int) -> torch.Tensor:
    """Return the NumPy-style Gaussian weight result using torch math."""

    def f(value: int, midpoint: float, var: float = 0.01) -> float:
        return math.exp(
            -((value - midpoint) * (value - midpoint))
            / (tile_width * tile_width)
            / (2 * var)
        ) / math.sqrt(2 * math.pi * var)

    x_probs = [f(x, (tile_width - 1) / 2) for x in range(tile_width)]
    y_probs = [f(y, tile_height / 2) for y in range(tile_height)]
    return torch.tensor(
        [[y_prob * x_prob for x_prob in x_probs] for y_prob in y_probs],
        dtype=torch.float32,
    )
