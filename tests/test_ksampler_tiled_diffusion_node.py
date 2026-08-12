# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the unified KSampler tiled diffusion ComfyUI node."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.nodes_v3.ksampler_tiled_diffusion import (
    KSamplerTiledDiffusionV3,
)
from simple_syrup.runtime import sampling_samplers, sampling_schedulers


def test_input_types_match_tiled_diffusion_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node exposes KSampler-style inputs plus mode and tile controls."""

    monkeypatch.setattr(sampling_samplers, "available_samplers", lambda: ("euler",))
    monkeypatch.setattr(
        sampling_schedulers,
        "available_schedulers",
        lambda: ("normal",),
    )
    schema = KSamplerTiledDiffusionV3.define_schema()
    required = {value.id: value for value in schema.inputs if not value.optional}
    optional = {value.id: value for value in schema.inputs if value.optional}

    assert tuple(required) == (
        "model",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "denoise",
        "diffusion_mode",
        "latent_tile_width",
        "latent_tile_height",
        "latent_tile_overlap",
        "latent_tile_batch_size",
    )
    assert required["diffusion_mode"].options == [
        "multidiffusion",
        "mixture_of_diffusers",
    ]
    assert required["diffusion_mode"].default == "multidiffusion"
    assert required["positive"].io_type == "CONDITIONING,CONDITIONING_BATCH"
    assert required["negative"].io_type == "CONDITIONING,CONDITIONING_BATCH"
    assert required["latent_tile_width"].default == 128
    assert required["latent_tile_width"].max == 512
    assert required["latent_tile_height"].default == 128
    assert required["latent_tile_height"].max == 512
    assert required["latent_tile_overlap"].default == 16
    assert required["latent_tile_batch_size"].default == 4
    assert optional["segs"].io_type == "SEGS"
    assert optional["region_masks"].io_type == "MASK"
    assert optional["regional_prompt_weight"].default == 0.5
    assert optional["region_mask_feather"].default == 0


def test_node_metadata_matches_contract() -> None:
    """The node declares the expected ComfyUI output contract."""

    schema = KSamplerTiledDiffusionV3.define_schema()

    assert schema.node_id == "SimpleSyrup.KSamplerTiledDiffusion"
    assert schema.display_name == "KSampler (Tiled Diffusion)"
    assert schema.category == "SimpleSyrup/Sampling"
    assert [(output.id, output.io_type) for output in schema.outputs] == [
        (None, "LATENT")
    ]


def test_sample_delegates_to_shared_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node remains thin and returns the service output unchanged."""

    fake_service = _FakeTiledDiffusionSamplingService()
    monkeypatch.setattr(
        KSamplerTiledDiffusionV3,
        "service_class",
        staticmethod(lambda: fake_service),
    )
    latent_image = {"samples": torch.zeros((1, 4, 4, 4))}

    (result,) = KSamplerTiledDiffusionV3.execute(
        model="model",
        seed=123,
        steps=20,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        positive="positive",
        negative="negative",
        latent_image=latent_image,
        denoise=0.8,
        diffusion_mode="mixture_of_diffusers",
        latent_tile_width=96,
        latent_tile_height=80,
        latent_tile_overlap=24,
        latent_tile_batch_size=3,
    )

    assert result is fake_service.output
    call = fake_service.calls[0]
    assert call["diffusion_mode"] == "mixture_of_diffusers"
    assert call["model"] == "model"
    assert call["seed"] == 123
    assert call["steps"] == 20
    assert call["cfg"] == 7.0
    assert call["sampler_name"] == "euler"
    assert call["scheduler"] == "normal"
    assert call["positive"] == "positive"
    assert call["negative"] == "negative"
    assert call["latent_image"] is latent_image
    assert call["denoise"] == 0.8
    assert call["latent_tile_width"] == 96
    assert call["latent_tile_height"] == 80
    assert call["latent_tile_overlap"] == 24
    assert call["latent_tile_batch_size"] == 3
    assert call["preview_context"] is None
    assert call["segs"] is None
    assert call["region_masks"] is None
    assert call["regional_prompt_weight"] == 0.5
    assert call["region_mask_feather"] == 0


def test_invalid_diffusion_mode_fails_before_runtime_sampling() -> None:
    """Unsupported modes are rejected before sampler side effects."""

    with pytest.raises(ValueError, match="diffusion_mode"):
        KSamplerTiledDiffusionV3.execute(
            model=object(),
            seed=123,
            steps=20,
            cfg=7.0,
            sampler_name="euler",
            scheduler="normal",
            positive=[],
            negative=[],
            latent_image={"samples": torch.zeros((1, 4, 4, 4))},
            denoise=0.8,
            diffusion_mode="full_latent",
            latent_tile_width=128,
            latent_tile_height=80,
            latent_tile_overlap=24,
            latent_tile_batch_size=3,
        )


class _FakeTiledDiffusionSamplingService:
    """Fake shared sampling service for node tests."""

    def __init__(self) -> None:
        """Create deterministic output and call records."""

        self.output: dict[str, Any] = {"samples": torch.ones((1, 4, 4, 4))}
        self.calls: list[dict[str, Any]] = []

    def sample(
        self,
        *,
        diffusion_mode: str,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: dict[str, Any],
        denoise: float,
        latent_tile_width: int,
        latent_tile_height: int,
        latent_tile_overlap: int,
        latent_tile_batch_size: int,
        preview_context: Any | None = None,
        segs: object | None = None,
        region_masks: object | None = None,
        regional_prompt_weight: float = 0.5,
        region_mask_feather: int = 0,
    ) -> dict[str, Any]:
        """Record sampling arguments and return a fixed latent."""

        self.calls.append(
            {
                "diffusion_mode": diffusion_mode,
                "model": model,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "positive": positive,
                "negative": negative,
                "latent_image": latent_image,
                "denoise": denoise,
                "latent_tile_width": latent_tile_width,
                "latent_tile_height": latent_tile_height,
                "latent_tile_overlap": latent_tile_overlap,
                "latent_tile_batch_size": latent_tile_batch_size,
                "preview_context": preview_context,
                "segs": segs,
                "region_masks": region_masks,
                "regional_prompt_weight": regional_prompt_weight,
                "region_mask_feather": region_mask_feather,
            }
        )
        return self.output
