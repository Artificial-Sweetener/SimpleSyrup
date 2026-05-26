# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the unified KSampler tiled diffusion ComfyUI node."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.nodes.ksampler_tiled_diffusion import KSamplerTiledDiffusion
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
    required = KSamplerTiledDiffusion.INPUT_TYPES()["required"]

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
    assert required["diffusion_mode"][0] == [
        "multidiffusion",
        "mixture_of_diffusers",
    ]
    assert required["diffusion_mode"][1]["default"] == "multidiffusion"
    assert required["positive"][0] == "CONDITIONING,CONDITIONING_BATCH"
    assert required["negative"][0] == "CONDITIONING,CONDITIONING_BATCH"
    assert required["latent_tile_width"][1]["default"] == 128
    assert required["latent_tile_width"][1]["max"] == 512
    assert required["latent_tile_height"][1]["default"] == 128
    assert required["latent_tile_height"][1]["max"] == 512
    assert required["latent_tile_overlap"][1]["default"] == 16
    assert required["latent_tile_batch_size"][1]["default"] == 4


def test_node_metadata_matches_contract() -> None:
    """The node declares the expected ComfyUI output contract."""

    assert KSamplerTiledDiffusion.RETURN_TYPES == ("LATENT",)
    assert KSamplerTiledDiffusion.FUNCTION == "sample"
    assert KSamplerTiledDiffusion.CATEGORY == "SimpleSyrup/Sampling"
    assert not hasattr(KSamplerTiledDiffusion, "RETURN_NAMES")


def test_sample_delegates_to_shared_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node remains thin and returns the service output unchanged."""

    fake_service = _FakeTiledDiffusionSamplingService()
    monkeypatch.setattr(
        KSamplerTiledDiffusion,
        "service_class",
        staticmethod(lambda: fake_service),
    )
    latent_image = {"samples": torch.zeros((1, 4, 4, 4))}

    (result,) = KSamplerTiledDiffusion().sample(
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


def test_invalid_diffusion_mode_fails_before_runtime_sampling() -> None:
    """Unsupported modes are rejected before sampler side effects."""

    with pytest.raises(ValueError, match="diffusion_mode"):
        KSamplerTiledDiffusion().sample(
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
            }
        )
        return self.output
