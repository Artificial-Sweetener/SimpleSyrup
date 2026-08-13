# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Select one tiled diffusion runtime for one latent sampling request."""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias

from ..domain.regional_features import RegionalCapabilityAdmission
from ..domain.tiled_diffusion import TiledDiffusionPlan
from ..runtime import mixture_of_diffusers_sampling, multidiffusion_sampling
from ..runtime.detail_previews import DetailPreviewContext

Latent: TypeAlias = dict[str, Any]


class TiledDiffusionItemSampler(Protocol):
    """Sample one latent item through a supplied tiled diffusion plan."""

    def __call__(
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
        preview_context: DetailPreviewContext | None,
        differential_diffusion: bool,
        capability_admission: RegionalCapabilityAdmission,
        tiled_plan: TiledDiffusionPlan | None = None,
    ) -> Latent:
        """Return one sampled latent item."""


class TiledDiffusionItemSamplingService:
    """Route one latent and optional tile plan to the selected runtime."""

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
        preview_context: DetailPreviewContext | None,
        differential_diffusion: bool,
        capability_admission: RegionalCapabilityAdmission,
        tiled_plan: TiledDiffusionPlan | None = None,
    ) -> Latent:
        """Invoke exactly one runtime with the unchanged sampling request."""

        sampler = (
            multidiffusion_sampling.sample_multidiffusion
            if diffusion_mode == "multidiffusion"
            else mixture_of_diffusers_sampling.sample_mixture_of_diffusers
        )
        return sampler(
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
            capability_admission=capability_admission,
            tiled_plan=tiled_plan,
        )
