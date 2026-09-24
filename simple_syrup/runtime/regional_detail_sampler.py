# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt shared detail sampling to regional MultiDiffusion runtime calls."""

from __future__ import annotations

from typing import Any

import torch

from ..domain.regional_detailing import LatentRegion
from . import regional_multidiffusion_sampling
from .detail_previews import DetailPreviewContext
from .detail_sampling import DetailSampler, Latent


class RegionalDetailSampler:
    """Adapt shared detail sampling helpers to regional MultiDiffusion."""

    def __init__(self, detail_sampler: DetailSampler | None = None) -> None:
        """Create the runtime adapter with injectable encode/decode behavior."""

        self._detail_sampler = detail_sampler or DetailSampler()

    def encode(self, vae: Any, pixels: torch.Tensor, tiled: bool) -> Latent:
        """Encode pixels into a latent dictionary."""

        return self._detail_sampler.encode(vae, pixels, tiled)

    def decode(self, vae: Any, latent: Latent, tiled: bool) -> torch.Tensor:
        """Decode latent samples into pixels."""

        return self._detail_sampler.decode(vae, latent, tiled)

    def sample_regions(
        self,
        *,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: Latent,
        regions: tuple[LatentRegion, ...],
        denoise: float,
        global_prompt_weight: float,
        preview_context: DetailPreviewContext | None = None,
        differential_diffusion: bool = False,
    ) -> Latent:
        """Sample one full latent with regional MultiDiffusion."""

        return regional_multidiffusion_sampling.sample_regional_multidiffusion(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            regions=regions,
            denoise=denoise,
            global_prompt_weight=global_prompt_weight,
            preview_context=preview_context,
            differential_diffusion=differential_diffusion,
        )
