# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for selectable tiled diffusion sampling."""

from __future__ import annotations

from typing import Any, ClassVar

from ..domain.tiled_diffusion import TILED_DIFFUSION_MODES
from ..runtime import sampling_samplers, sampling_schedulers
from ..services.tiled_diffusion_sampling_service import TiledDiffusionSamplingService
from . import tooltips

Latent = dict[str, Any]
MAX_LATENT_TILE_SIZE = 512


class KSamplerTiledDiffusion:
    """Sample latents with selectable tiled diffusion denoising."""

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = (tooltips.DENOISED_LATENT_OUTPUT,)
    FUNCTION = "sample"
    CATEGORY = "SimpleSyrup/Sampling"
    DESCRIPTION = "Denoises latents with selectable tiled diffusion sampling."
    SEARCH_ALIASES = [
        "ksampler",
        "sampler",
        "tiled diffusion",
        "multidiffusion",
        "multi diffusion",
        "mixture of diffusers",
    ]

    service_class: ClassVar[type[TiledDiffusionSamplingService]] = (
        TiledDiffusionSamplingService
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare ComfyUI inputs for selectable tiled diffusion sampling."""

        return {
            "required": {
                "model": ("MODEL", {"tooltip": tooltips.SAMPLING_MODEL}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": tooltips.SAMPLING_SEED,
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 10000,
                        "tooltip": tooltips.SAMPLING_STEPS,
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": tooltips.SAMPLING_CFG,
                    },
                ),
                "sampler_name": (
                    sampling_samplers.available_samplers(),
                    {"tooltip": tooltips.SAMPLER_NAME},
                ),
                "scheduler": (
                    sampling_schedulers.available_schedulers(),
                    {"tooltip": tooltips.SCHEDULER},
                ),
                "positive": (
                    "CONDITIONING,CONDITIONING_BATCH",
                    {"tooltip": tooltips.POSITIVE_CONDITIONING},
                ),
                "negative": (
                    "CONDITIONING,CONDITIONING_BATCH",
                    {"tooltip": tooltips.NEGATIVE_CONDITIONING},
                ),
                "latent_image": ("LATENT", {"tooltip": tooltips.LATENT_IMAGE}),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": tooltips.DENOISE_STRENGTH,
                    },
                ),
                "diffusion_mode": (
                    list(TILED_DIFFUSION_MODES),
                    {
                        "default": "multidiffusion",
                        "tooltip": (
                            "Tiled sampling blend method. MultiDiffusion is steady; "
                            "Mixture of Diffusers can blend tile predictions more "
                            "softly."
                        ),
                    },
                ),
                "latent_tile_width": (
                    "INT",
                    {
                        "default": 128,
                        "min": 16,
                        "max": MAX_LATENT_TILE_SIZE,
                        "step": 16,
                        "tooltip": tooltips.LATENT_TILE_WIDTH,
                    },
                ),
                "latent_tile_height": (
                    "INT",
                    {
                        "default": 128,
                        "min": 16,
                        "max": MAX_LATENT_TILE_SIZE,
                        "step": 16,
                        "tooltip": tooltips.LATENT_TILE_HEIGHT,
                    },
                ),
                "latent_tile_overlap": (
                    "INT",
                    {
                        "default": 16,
                        "min": 0,
                        "max": 256,
                        "step": 4,
                        "tooltip": tooltips.LATENT_TILE_OVERLAP,
                    },
                ),
                "latent_tile_batch_size": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": tooltips.LATENT_TILE_BATCH_SIZE,
                    },
                ),
            }
        }

    def sample(
        self,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: Latent,
        denoise: float = 1.0,
        diffusion_mode: str = "multidiffusion",
        latent_tile_width: int = 128,
        latent_tile_height: int = 128,
        latent_tile_overlap: int = 16,
        latent_tile_batch_size: int = 4,
    ) -> tuple[Latent]:
        """Sample a latent with the selected tiled diffusion method."""

        output = self.service_class().sample(
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
            preview_context=None,
        )
        return (output,)
