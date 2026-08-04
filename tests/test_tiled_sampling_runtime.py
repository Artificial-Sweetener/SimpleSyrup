# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for shared tiled sampling runtime helpers."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.contextual_diffusion import SpatialContext
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

    with pytest.warns(UserWarning, match="nested tensors.*prototype stage"):
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


def test_tile_conditioning_repeats_complete_reference_latents() -> None:
    """Independent reference images remain complete for every tiled model call."""

    canvas_sized_reference = torch.arange(
        1 * 4 * 4 * 8,
        dtype=torch.float32,
    ).reshape((1, 4, 4, 8))
    smaller_reference = torch.full((1, 4, 3, 5), 7.0)
    tiles = (LatentTile(0, 0, 4, 4), LatentTile(4, 0, 4, 4))

    transformed = tiled_sampling.tile_conditioning(
        conditioning={
            "ref_latents": [canvas_sized_reference, smaller_reference],
            "ref_latents_method": "index",
        },
        tiles=tiles,
        input_batch_size=1,
        latent_height=4,
        latent_width=8,
        tiled_timestep=torch.tensor([1.0, 1.0]),
    )

    references = transformed["ref_latents"]
    assert isinstance(references, list)
    assert torch.equal(
        references[0],
        torch.cat([canvas_sized_reference, canvas_sized_reference], dim=0),
    )
    assert torch.equal(
        references[1],
        torch.cat([smaller_reference, smaller_reference], dim=0),
    )
    assert transformed["ref_latents_method"] == "index"


def test_spatial_context_conditioning_repeats_complete_reference_latents() -> None:
    """Global and semantic contexts share complete independent reference images."""

    reference = torch.arange(1 * 4 * 8 * 12, dtype=torch.float32).reshape((1, 4, 8, 12))
    contexts = (
        SpatialContext(0, 0, 12, 8, 6, 4),
        SpatialContext(2, 1, 8, 6, 6, 4),
    )

    transformed = tiled_sampling.spatial_context_conditioning(
        conditioning={"ref_latents": [reference]},
        contexts=contexts,
        input_batch_size=1,
        latent_height=8,
        latent_width=12,
        context_timestep=torch.tensor([1.0, 1.0]),
    )

    references = transformed["ref_latents"]
    assert isinstance(references, list)
    assert torch.equal(references[0], torch.cat([reference, reference], dim=0))


def test_reference_latents_reject_ambiguous_batch_alignment() -> None:
    """Reference batches must align explicitly with each model input batch."""

    with pytest.raises(ValueError, match="ref_latents tensor batch size"):
        tiled_sampling.tile_conditioning(
            conditioning={"ref_latents": [torch.zeros((3, 4, 8, 12))]},
            tiles=(LatentTile(0, 0, 6, 8), LatentTile(6, 0, 6, 8)),
            input_batch_size=2,
            latent_height=8,
            latent_width=12,
            tiled_timestep=torch.ones((4,)),
        )


def test_spatial_context_args_resize_latent_and_canvas_conditioning() -> None:
    """A global context resizes spatial tensors and repeats aligned metadata."""

    x = torch.arange(2 * 1 * 8 * 12, dtype=torch.float32).reshape((2, 1, 8, 12))
    timestep = torch.tensor([0.5, 0.75])
    context = SpatialContext(0, 0, 12, 8, 6, 4)

    transformed = tiled_sampling.make_spatial_context_model_args(
        args={
            "input": x,
            "timestep": timestep,
            "cond_or_uncond": [0, 1],
            "c": {
                "c_concat": x.clone(),
                "c_crossattn": torch.ones((2, 3, 1)),
                "transformer_options": {"cond_or_uncond": [0, 1]},
            },
        },
        contexts=(context,),
        input_batch_size=2,
        latent_height=8,
        latent_width=12,
    )

    assert transformed["input"].shape == (2, 1, 4, 6)
    assert transformed["c"]["c_concat"].shape == (2, 1, 4, 6)
    assert transformed["c"]["c_crossattn"].shape == (2, 3, 1)
    assert transformed["c"]["transformer_options"]["cond_or_uncond"] == [0, 1]


def test_spatial_context_args_batch_equal_shapes_for_5d_latents() -> None:
    """Contexts batch across the spatial axes of singleton-depth latents."""

    x = torch.zeros((1, 16, 1, 8, 16))
    contexts = (
        SpatialContext(0, 0, 8, 8, 8, 8),
        SpatialContext(8, 0, 8, 8, 8, 8),
    )

    transformed = tiled_sampling.make_spatial_context_model_args(
        args={"input": x, "timestep": torch.tensor([1.0]), "c": {}},
        contexts=contexts,
        input_batch_size=1,
        latent_height=8,
        latent_width=16,
    )

    assert transformed["input"].shape == (2, 16, 1, 8, 8)
    assert transformed["timestep"].shape == (2,)


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


def test_semantic_tile_weight_cache_reuses_resident_weights() -> None:
    """Semantic tile weights are materialized once for repeated model outputs."""

    tile = LatentTile(
        x=0,
        y=0,
        width=4,
        height=4,
        weight_mask=torch.ones((4, 4), dtype=torch.float32),
    )
    cache = tiled_sampling.SemanticTileWeightCache((tile,))
    output = torch.zeros((1, 4, 4, 4), dtype=torch.float16)

    first = cache.for_output(output)
    second = cache.for_output(output)
    model_weight, accumulation_weight = cache.for_tile(first, tile)

    assert first is second
    assert model_weight is first.model[0]
    assert accumulation_weight is first.accumulation[0]
    assert model_weight.dtype == torch.float16
    assert accumulation_weight.dtype == torch.float32


def test_tile_prediction_accumulator_selects_multidiffusion_or_mod_weights() -> None:
    """One shared accumulator preserves the distinct overlap blending policies."""

    plan = build_tiled_diffusion_plan(6, 4, 4, 4, 2, 2)
    x = torch.arange(6, dtype=torch.float32).reshape(1, 1, 1, 6).expand(1, 1, 4, 6)
    args = {"input": x, "timestep": torch.tensor([1.0]), "c": {}}

    def evaluate(tiled_args: dict[str, object]) -> torch.Tensor:
        """Return a distinct constant prediction for each source tile."""

        tiled_input = tiled_args["input"]
        if not isinstance(tiled_input, torch.Tensor):
            raise TypeError("Test evaluator expected a tiled input tensor.")
        means = tiled_input.mean(dim=(-2, -1), keepdim=True)
        return means.expand_as(tiled_input)

    multidiffusion = tiled_sampling.TilePredictionAccumulator(
        plan,
        diffusion_mode="multidiffusion",
    ).predict(args=args, x=x, evaluate=evaluate)
    mixture = tiled_sampling.TilePredictionAccumulator(
        plan,
        diffusion_mode="mixture_of_diffusers",
    ).predict(args=args, x=x, evaluate=evaluate)

    assert torch.allclose(multidiffusion[:, :, :, 2:4], torch.full((1, 1, 4, 2), 2.5))
    assert not torch.allclose(mixture[:, :, :, 2:4], multidiffusion[:, :, :, 2:4])
    assert torch.allclose(mixture[:, :, :, :2], multidiffusion[:, :, :, :2])
    assert torch.allclose(mixture[:, :, :, 4:], multidiffusion[:, :, :, 4:])


def test_contains_unsupported_conditioning_key_finds_nested_values() -> None:
    """Unsupported regional and control keys are detected recursively."""

    conditioning = [{"model_conds": {"nested": [{"mask": torch.ones((1, 1))}]}}]

    assert tiled_sampling.contains_unsupported_conditioning_key(conditioning)


def test_full_context_masks_require_explicit_tiled_support() -> None:
    """Only explicit non-cropped masks pass the regional tiled policy."""

    supported = [
        ["tensor", {"mask": torch.ones((1, 2, 2)), "set_area_to_bounds": False}]
    ]
    cropped = [["tensor", {"mask": torch.ones((1, 2, 2)), "set_area_to_bounds": True}]]

    assert tiled_sampling.contains_unsupported_conditioning_key(supported)
    assert not tiled_sampling.contains_unsupported_conditioning_key(
        supported,
        allow_full_context_masks=True,
    )
    assert tiled_sampling.contains_unsupported_conditioning_key(
        cropped,
        allow_full_context_masks=True,
    )
