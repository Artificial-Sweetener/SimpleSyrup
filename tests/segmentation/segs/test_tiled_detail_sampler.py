# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify tiled detail sampling adapter delegation."""

from __future__ import annotations

from typing import Any

import torch

from simple_syrup.domain.segs import CropRegion
from simple_syrup.runtime.detail_previews import DetailPreviewContext
from simple_syrup.runtime.detail_sampling import Latent
from simple_syrup.services.tiled_detail_sampler import TiledDetailSampler


def test_tiled_detail_sampler_delegates_to_shared_sampling_service() -> None:
    """The detailer adapter uses the shared tiled diffusion dispatch service."""

    tiled_sampling_service = _FakeTiledSamplingService()
    latent = {"samples": torch.zeros((1, 4, 4, 4))}
    preview_context = DetailPreviewContext(
        image=_image(),
        work_region=CropRegion(2, 2, 6, 6),
        work_mask=torch.ones((4, 4)),
    )
    result = TiledDetailSampler(
        tiled_sampling_service=tiled_sampling_service
    ).sample_tiled(
        diffusion_mode="mixture_of_diffusers",
        model="model",
        seed=123,
        steps=4,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        positive="positive",
        negative="negative",
        latent_image=latent,
        denoise=0.5,
        latent_tile_width=128,
        latent_tile_height=80,
        latent_tile_overlap=12,
        latent_tile_batch_size=3,
        preview_context=preview_context,
        differential_diffusion=True,
    )
    assert result is latent
    call = tiled_sampling_service.calls[0]
    assert call["diffusion_mode"] == "mixture_of_diffusers"
    assert call["latent_image"] is latent
    assert call["preview_context"] is preview_context
    assert call["differential_diffusion"] is True


class _FakeTiledSamplingService:
    """Record calls to the shared tiled diffusion sampling service."""

    def __init__(self) -> None:
        """Create empty call records."""

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
        latent_image: Latent,
        denoise: float,
        latent_tile_width: int,
        latent_tile_height: int,
        latent_tile_overlap: int,
        latent_tile_batch_size: int,
        preview_context: DetailPreviewContext | None = None,
        differential_diffusion: bool = False,
    ) -> Latent:
        """Record tiled sampling arguments and return the latent unchanged."""

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
                "differential_diffusion": differential_diffusion,
            }
        )
        return latent_image


def _image() -> torch.Tensor:
    """Return a small preview image."""

    return torch.zeros((1, 8, 8, 3), dtype=torch.float32)
