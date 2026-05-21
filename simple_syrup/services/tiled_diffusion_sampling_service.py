# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application service for selectable tiled diffusion latent sampling."""

from __future__ import annotations

from typing import Any

from ..domain.tiled_diffusion import validate_tiled_diffusion_mode
from ..runtime import mixture_of_diffusers_sampling, multidiffusion_sampling
from ..runtime.detail_previews import DetailPreviewContext

Latent = dict[str, Any]


class TiledDiffusionSamplingService:
    """Route tiled diffusion sampling requests to the selected runtime."""

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
    ) -> Latent:
        """Sample a latent with the selected tiled diffusion method."""

        validate_tiled_diffusion_mode(diffusion_mode)
        if diffusion_mode == "multidiffusion":
            return multidiffusion_sampling.sample_multidiffusion(
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
            )
        return mixture_of_diffusers_sampling.sample_mixture_of_diffusers(
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
        )
