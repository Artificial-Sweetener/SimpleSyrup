# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for full-latent and tiled regional KSampler v3 nodes."""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from simple_syrup.nodes_v3.ksampler_prompt_by_region import (
    KSamplerPromptByRegionV3,
)
from simple_syrup.nodes_v3.ksampler_prompt_by_tiled_region import (
    KSamplerPromptByTiledRegionV3,
)


class FakeConditioningService:
    """Record regional assembly and return recognizable conditioning."""

    calls: ClassVar[list[dict[str, object]]] = []

    def assemble(self, **kwargs: object) -> tuple[object, object]:
        """Record one shared assembly request."""

        type(self).calls.append(kwargs)
        return "assembled-positive", "assembled-negative"


class FakeSamplingService:
    """Record normal or tiled sampling arguments."""

    calls: ClassVar[list[dict[str, Any]]] = []
    output: ClassVar[dict[str, Any]] = {"samples": torch.ones((1, 4, 2, 2))}

    def sample(self, **kwargs: Any) -> dict[str, Any]:
        """Record and return one latent."""

        type(self).calls.append(kwargs)
        return self.output


def test_schemas_expose_exact_names_and_shared_regional_contract() -> None:
    """Both nodes expose global-first regional inputs with only tiled extras."""

    normal = KSamplerPromptByRegionV3.define_schema()
    tiled = KSamplerPromptByTiledRegionV3.define_schema()
    normal_ids = [value.id for value in normal.inputs]
    tiled_ids = [value.id for value in tiled.inputs]

    assert normal.node_id == "SimpleSyrup.KSamplerPromptByRegion"
    assert normal.display_name == "KSampler (Prompt by Region)"
    assert tiled.node_id == "SimpleSyrup.KSamplerPromptByTiledRegion"
    assert tiled.display_name == "KSampler (Prompt by Tiled Region)"
    assert normal_ids == [
        "model",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "region_masks",
        "regional_prompt_weight",
        "region_mask_feather",
        "latent_image",
        "denoise",
    ]
    assert tiled_ids[: len(normal_ids)] == normal_ids
    assert tiled_ids[len(normal_ids) :] == [
        "diffusion_mode",
        "latent_tile_width",
        "latent_tile_height",
        "latent_tile_overlap",
        "latent_tile_batch_size",
    ]
    regional_weight = normal.inputs[9]
    assert regional_weight.default == 0.5
    assert regional_weight.min == 0.0
    assert regional_weight.max == 1.0
    assert [output.io_type for output in normal.outputs] == ["LATENT"]
    assert [output.io_type for output in tiled.outputs] == ["LATENT"]


def test_non_tiled_node_assembles_then_samples_full_latent() -> None:
    """The non-tiled node delegates regional and sampling concerns once."""

    _reset_fakes()
    original_conditioning = KSamplerPromptByRegionV3.conditioning_service_class
    original_sampling = KSamplerPromptByRegionV3.sampling_service_class
    KSamplerPromptByRegionV3.conditioning_service_class = FakeConditioningService  # type: ignore[assignment]
    KSamplerPromptByRegionV3.sampling_service_class = FakeSamplingService  # type: ignore[assignment]
    masks = torch.ones((2, 8, 8))
    latent = {"samples": torch.zeros((3, 4, 2, 2))}
    try:
        (output,) = KSamplerPromptByRegionV3.execute(
            model="model",
            seed=3,
            steps=10,
            cfg=5.0,
            sampler_name="euler",
            scheduler="normal",
            positive="positive-batch",
            negative="negative-batch",
            region_masks=masks,
            regional_prompt_weight=0.75,
            region_mask_feather=4,
            latent_image=latent,
            denoise=0.7,
        )
    finally:
        KSamplerPromptByRegionV3.conditioning_service_class = original_conditioning
        KSamplerPromptByRegionV3.sampling_service_class = original_sampling

    assert output is FakeSamplingService.output
    assert FakeConditioningService.calls == [
        {
            "positive": "positive-batch",
            "negative": "negative-batch",
            "masks": masks,
            "regional_prompt_weight": 0.75,
            "region_mask_feather": 4,
        }
    ]
    call = FakeSamplingService.calls[0]
    assert call["positive"] == "assembled-positive"
    assert call["negative"] == "assembled-negative"
    assert call["latent_image"] is latent
    assert "diffusion_mode" not in call


def test_tiled_node_enables_only_full_context_regional_masks() -> None:
    """The tiled node shares assembly and explicitly opts into mask support."""

    _reset_fakes()
    original_conditioning = KSamplerPromptByTiledRegionV3.conditioning_service_class
    original_sampling = KSamplerPromptByTiledRegionV3.sampling_service_class
    KSamplerPromptByTiledRegionV3.conditioning_service_class = FakeConditioningService  # type: ignore[assignment]
    KSamplerPromptByTiledRegionV3.sampling_service_class = FakeSamplingService  # type: ignore[assignment]
    masks = torch.ones((1, 8, 8))
    latent = {"samples": torch.zeros((1, 4, 16, 16))}
    try:
        (output,) = KSamplerPromptByTiledRegionV3.execute(
            model="model",
            seed=3,
            steps=10,
            cfg=5.0,
            sampler_name="euler",
            scheduler="normal",
            positive="positive-batch",
            negative="negative-batch",
            region_masks=masks,
            regional_prompt_weight=0.6,
            region_mask_feather=0,
            latent_image=latent,
            denoise=0.7,
            diffusion_mode="mixture_of_diffusers",
            latent_tile_width=8,
            latent_tile_height=8,
            latent_tile_overlap=2,
            latent_tile_batch_size=3,
        )
    finally:
        KSamplerPromptByTiledRegionV3.conditioning_service_class = original_conditioning
        KSamplerPromptByTiledRegionV3.sampling_service_class = original_sampling

    assert output is FakeSamplingService.output
    assert FakeConditioningService.calls == [
        {
            "positive": "positive-batch",
            "negative": "negative-batch",
            "masks": masks,
            "regional_prompt_weight": 0.6,
            "region_mask_feather": 0,
        }
    ]
    call = FakeSamplingService.calls[0]
    assert call["diffusion_mode"] == "mixture_of_diffusers"
    assert call["allow_full_context_masks"] is True
    assert call["latent_tile_width"] == 8
    assert call["latent_tile_height"] == 8
    assert call["latent_tile_overlap"] == 2
    assert call["latent_tile_batch_size"] == 3


def _reset_fakes() -> None:
    """Clear shared fake records between node tests."""

    FakeConditioningService.calls = []
    FakeSamplingService.calls = []
