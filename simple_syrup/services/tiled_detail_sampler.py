# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt detail encode/decode behavior to tiled diffusion sampling."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from ..runtime.detail_previews import DetailPreviewContext
from ..runtime.detail_sampling import DetailSampler, Latent
from .tiled_diffusion_sampling_service import TiledDiffusionSamplingService


class TiledDiffusionLatentSamplingBoundary(Protocol):
    """Latent sampling boundary for selectable tiled diffusion modes."""

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
        """Sample a latent using the selected tiled diffusion mode."""


class TiledDetailSampler:
    """Adapt shared VAE helpers and tiled diffusion application services."""

    def __init__(
        self,
        detail_sampler: DetailSampler | None = None,
        tiled_sampling_service: TiledDiffusionLatentSamplingBoundary | None = None,
    ) -> None:
        """Create the adapter with injectable sampling collaborators."""

        self._detail_sampler = detail_sampler or DetailSampler()
        self._tiled_sampling_service = (
            tiled_sampling_service or TiledDiffusionSamplingService()
        )

    def encode(self, vae: Any, pixels: torch.Tensor, tiled: bool) -> Latent:
        """Encode pixels into a latent dictionary."""

        return self._detail_sampler.encode(vae, pixels, tiled)

    def decode(self, vae: Any, latent: Latent, tiled: bool) -> torch.Tensor:
        """Decode latent samples into pixels."""

        return self._detail_sampler.decode(vae, latent, tiled)

    def sample_tiled(
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
        """Sample one latent crop with the selected tiled diffusion runtime."""

        return self._tiled_sampling_service.sample(
            diffusion_mode=diffusion_mode,
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
            latent_tile_width=latent_tile_width,
            latent_tile_height=latent_tile_height,
            latent_tile_overlap=latent_tile_overlap,
            latent_tile_batch_size=latent_tile_batch_size,
            preview_context=preview_context,
            differential_diffusion=differential_diffusion,
        )
