# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for shared tiled sampling runtime helpers."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.tiled_diffusion import LatentTile, build_tiled_diffusion_plan
from simple_syrup.runtime import tiled_sampling


def test_validate_latent_samples_accepts_bchw() -> None:
    """Standard BCHW latents are valid for tiled samplers."""

    samples = torch.zeros((1, 4, 8, 8))

    assert (
        tiled_sampling.validate_latent_samples(
            {"samples": samples},
            sampler_label="TestSampler",
        )
        is samples
    )


def test_validate_latent_samples_accepts_singleton_depth_bcdhw() -> None:
    """Singleton-depth BCDHW latents are valid for Anima-style models."""

    samples = torch.zeros((1, 16, 1, 8, 8))

    assert (
        tiled_sampling.validate_latent_samples(
            {"samples": samples},
            sampler_label="TestSampler",
        )
        is samples
    )


def test_validate_latent_samples_rejects_non_tensor() -> None:
    """Latent dictionaries must contain tensor samples."""

    with pytest.raises(ValueError, match="latent samples must be a torch tensor"):
        tiled_sampling.validate_latent_samples(
            {"samples": "not-a-tensor"},
            sampler_label="TestSampler",
        )


def test_validate_tensor_shape_rejects_nested_tensor() -> None:
    """Nested tensors are rejected before spatial tiling."""

    samples = torch.nested.nested_tensor([torch.zeros((4, 8, 8))])

    with pytest.raises(ValueError, match="non-nested latent samples"):
        tiled_sampling.validate_tensor_shape(samples, sampler_label="TestSampler")


def test_validate_tensor_shape_rejects_non_singleton_depth_5d() -> None:
    """Non-singleton depth 5D latents remain unsupported."""

    with pytest.raises(ValueError, match="singleton third axis"):
        tiled_sampling.validate_tensor_shape(
            torch.zeros((1, 16, 2, 8, 8)),
            sampler_label="TestSampler",
        )


def test_spatial_tile_slicer_crops_final_axes_for_4d() -> None:
    """Spatial slicers crop height and width for BCHW tensors."""

    tensor = torch.arange(1 * 1 * 4 * 6).reshape((1, 1, 4, 6))
    tile = LatentTile(x=2, y=1, width=3, height=2)

    cropped = tensor[tiled_sampling.spatial_tile_slicer(tile, tensor.ndim)]

    assert torch.equal(cropped, tensor[:, :, 1:3, 2:5])


def test_spatial_tile_slicer_crops_final_axes_for_5d() -> None:
    """Spatial slicers preserve singleton depth while cropping BCDHW tensors."""

    tensor = torch.arange(1 * 2 * 1 * 4 * 6).reshape((1, 2, 1, 4, 6))
    tile = LatentTile(x=1, y=2, width=4, height=2)

    cropped = tensor[tiled_sampling.spatial_tile_slicer(tile, tensor.ndim)]

    assert torch.equal(cropped, tensor[:, :, :, 2:4, 1:5])


def test_tile_tensor_crops_spatial_tensor_per_tile() -> None:
    """Spatial conditioning tensors are cropped and concatenated per tile."""

    tensor = torch.arange(2 * 1 * 4 * 8, dtype=torch.float32).reshape((2, 1, 4, 8))
    tiles = (LatentTile(0, 0, 4, 4), LatentTile(4, 0, 4, 4))

    tiled = tiled_sampling.tile_tensor(
        tensor,
        tiles=tiles,
        input_batch_size=2,
        latent_height=4,
        latent_width=8,
    )

    assert tiled.shape == (4, 1, 4, 4)
    assert torch.equal(tiled[:2], tensor[:, :, :, :4])
    assert torch.equal(tiled[2:], tensor[:, :, :, 4:])


def test_tile_tensor_repeats_matching_batch_tensor() -> None:
    """Batch-aligned non-spatial tensors repeat once per tile."""

    tensor = torch.tensor([[1.0], [2.0]])
    tiles = (LatentTile(0, 0, 4, 4), LatentTile(4, 0, 4, 4))

    tiled = tiled_sampling.tile_tensor(
        tensor,
        tiles=tiles,
        input_batch_size=2,
        latent_height=4,
        latent_width=8,
    )

    assert torch.equal(tiled, torch.tensor([[1.0], [2.0], [1.0], [2.0]]))


def test_tile_tensor_repeats_singleton_batch_to_tiled_batch_size() -> None:
    """Singleton-batch tensors expand to the full tiled input batch size."""

    tensor = torch.tensor([[5.0, 6.0]])
    tiles = (LatentTile(0, 0, 4, 4), LatentTile(4, 0, 4, 4))

    tiled = tiled_sampling.tile_tensor(
        tensor,
        tiles=tiles,
        input_batch_size=2,
        latent_height=4,
        latent_width=8,
    )

    assert tiled.shape == (4, 2)
    assert torch.equal(tiled, tensor.repeat((4, 1)))


def test_tile_transformer_options_repeats_model_metadata() -> None:
    """Transformer metadata aligned to model batches repeats per tile."""

    timestep = torch.tensor([0.5, 0.75, 0.5, 0.75])
    options = {
        "cond_or_uncond": [0, 1],
        "uuids": ("positive", "negative"),
        "sigmas": torch.tensor([1.0, 0.0]),
        "sample_sigmas": torch.tensor([1.0, 0.0]),
    }

    tiled = tiled_sampling.tile_transformer_options(
        options,
        tile_count=2,
        tiled_timestep=timestep,
    )

    assert tiled["cond_or_uncond"] == [0, 1, 0, 1]
    assert tiled["uuids"] == ("positive", "negative", "positive", "negative")
    assert torch.equal(tiled["sigmas"], timestep)
    assert torch.equal(tiled["sample_sigmas"], torch.tensor([1.0, 0.0]))


def test_new_spatial_weight_buffer_broadcasts_over_spatial_axes() -> None:
    """Spatial weight buffers broadcast over BCHW and BCDHW model outputs."""

    plan = build_tiled_diffusion_plan(8, 4, 4, 4, 0, 1)

    assert tiled_sampling.new_spatial_weight_buffer(
        torch.zeros((2, 4, 4, 8)),
        plan,
    ).shape == (1, 1, 4, 8)
    assert tiled_sampling.new_spatial_weight_buffer(
        torch.zeros((2, 16, 1, 4, 8)),
        plan,
    ).shape == (1, 1, 1, 4, 8)


def test_contains_unsupported_conditioning_key_finds_nested_values() -> None:
    """Unsupported regional and control keys are detected recursively."""

    conditioning = [{"model_conds": {"nested": [{"mask": torch.ones((1, 1))}]}}]

    assert tiled_sampling.contains_unsupported_conditioning_key(conditioning)
