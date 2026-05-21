# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the tiled diffusion scale-factor detailer node contract."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from simple_syrup.domain.segs import BoundingBox, CropRegion, NativeSegs, Segment
from simple_syrup.nodes.detail_segs_by_scale_factor_tiled_diffusion import (
    DetailSEGSByScaleFactorTiledDiffusion,
)
from simple_syrup.runtime import sampling_samplers, sampling_schedulers
from simple_syrup.services.detail_segs_by_scale_factor_tiled_diffusion_service import (
    TiledDetailerResult,
)


def test_tiled_detailer_node_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tiled detailer node exposes its planned ComfyUI contract."""

    monkeypatch.setattr(sampling_samplers, "available_samplers", lambda: ("euler",))
    monkeypatch.setattr(
        sampling_schedulers,
        "available_schedulers",
        lambda: ("normal",),
    )
    inputs = DetailSEGSByScaleFactorTiledDiffusion.INPUT_TYPES()

    assert DetailSEGSByScaleFactorTiledDiffusion.RETURN_TYPES == ("IMAGE",)
    assert DetailSEGSByScaleFactorTiledDiffusion.RETURN_NAMES == ("image",)
    assert DetailSEGSByScaleFactorTiledDiffusion.INPUT_IS_LIST is True
    assert DetailSEGSByScaleFactorTiledDiffusion.CATEGORY == "SimpleSyrup/Detailing"
    assert list(inputs["required"]) == [
        "image",
        "segs",
        "model",
        "vae",
        "positive",
        "negative",
        "scale_factor",
        "upscale_method",
        "clamp_size",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "denoise",
        "feather",
        "noise_mask",
        "noise_mask_feather",
        "tiled_encode",
        "tiled_decode",
        "diffusion_mode",
        "latent_tile_width",
        "latent_tile_height",
        "latent_tile_overlap",
        "latent_tile_batch_size",
    ]
    assert inputs["required"]["diffusion_mode"][0] == [
        "multidiffusion",
        "mixture_of_diffusers",
    ]
    scale_factor_options = inputs["required"]["scale_factor"][1]
    assert scale_factor_options["default"] == 1.5
    assert scale_factor_options["min"] == 1.0
    assert scale_factor_options["max"] == 5.0
    assert scale_factor_options["step"] == 0.1
    assert inputs["required"]["diffusion_mode"][1]["default"] == "multidiffusion"
    assert inputs["required"]["latent_tile_width"][1]["default"] == 128
    assert inputs["required"]["latent_tile_batch_size"][1]["default"] == 4


def test_tiled_detailer_node_delegates_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tiled detailer forwards normalized values to its service."""

    fake_service = _FakeTiledDetailerService()
    monkeypatch.setattr(
        DetailSEGSByScaleFactorTiledDiffusion,
        "service_class",
        staticmethod(lambda: fake_service),
    )
    image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)

    (output_image,) = DetailSEGSByScaleFactorTiledDiffusion().detail(
        image,
        _segs(),
        model="model",
        vae="vae",
        positive=["positive"],
        negative=["negative"],
        scale_factor=1.5,
        upscale_method="bicubic",
        clamp_size=0,
        seed=1,
        steps=2,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        denoise=0.5,
        feather=5,
        noise_mask=True,
        noise_mask_feather=20,
        tiled_encode=False,
        tiled_decode=True,
        diffusion_mode="mixture_of_diffusers",
        latent_tile_width=64,
        latent_tile_height=80,
        latent_tile_overlap=12,
        latent_tile_batch_size=3,
    )

    assert torch.equal(cast(torch.Tensor, output_image), image + 1.0)
    call = fake_service.calls[0]
    assert call["upscale_method"] == "bicubic"
    assert call["diffusion_mode"] == "mixture_of_diffusers"
    assert call["latent_tile_width"] == 64
    assert call["latent_tile_height"] == 80
    assert call["latent_tile_overlap"] == 12
    assert call["latent_tile_batch_size"] == 3


class _FakeTiledDetailerService:
    """Fake tiled detailer service for node tests."""

    def __init__(self) -> None:
        """Create a call-recording fake service."""

        self.calls: list[dict[str, object]] = []

    def detail(
        self,
        image: object,
        segs: object,
        model: Any,
        vae: Any,
        positive: Any,
        negative: Any,
        scale_factor: float,
        upscale_method: str,
        clamp_size: int,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        feather: int,
        noise_mask: bool,
        noise_mask_feather: int,
        tiled_encode: bool,
        tiled_decode: bool,
        diffusion_mode: str,
        latent_tile_width: int,
        latent_tile_height: int,
        latent_tile_overlap: int,
        latent_tile_batch_size: int,
    ) -> TiledDetailerResult:
        """Return deterministic output and record tiled detailer inputs."""

        self.calls.append(
            {
                "image": image,
                "segs": segs,
                "model": model,
                "vae": vae,
                "positive": positive,
                "negative": negative,
                "scale_factor": scale_factor,
                "upscale_method": upscale_method,
                "clamp_size": clamp_size,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "feather": feather,
                "noise_mask": noise_mask,
                "noise_mask_feather": noise_mask_feather,
                "tiled_encode": tiled_encode,
                "tiled_decode": tiled_decode,
                "diffusion_mode": diffusion_mode,
                "latent_tile_width": latent_tile_width,
                "latent_tile_height": latent_tile_height,
                "latent_tile_overlap": latent_tile_overlap,
                "latent_tile_batch_size": latent_tile_batch_size,
            }
        )
        return TiledDetailerResult(image=cast(torch.Tensor, image) + 1.0)


def _segs() -> NativeSegs:
    """Return native SEGS for node tests."""

    segment = Segment(
        cropped_image=None,
        cropped_mask=torch.ones((2, 2)),
        confidence=1.0,
        crop_region=CropRegion(0, 0, 2, 2),
        bbox=BoundingBox(0, 0, 2, 2),
        label="face",
    )
    return (8, 8), (segment,)
