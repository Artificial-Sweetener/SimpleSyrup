# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for KSampler Extras."""

from __future__ import annotations

from typing import Any, ClassVar

from ..runtime import sampling_samplers, sampling_schedulers
from ..services.ksampler_sampling_service import KSamplerSamplingService
from . import tooltips

Latent = dict[str, Any]


class KSamplerExtras:
    """Expose KSampler-style sampling with extended scheduler options."""

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = (tooltips.DENOISED_LATENT_OUTPUT,)
    FUNCTION = "sample"
    CATEGORY = "SimpleSyrup/Sampling"
    DESCRIPTION = (
        "Denoises latents with extended sampler and scheduler options for "
        "compatible workflows."
    )
    SEARCH_ALIASES = ["ksampler", "sampler", "ays", "gits", "lcm"]
    service_class: ClassVar[type[KSamplerSamplingService]] = KSamplerSamplingService

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare ComfyUI inputs for extra scheduler sampling."""

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
    ) -> tuple[Latent]:
        """Sample a latent with ComfyUI samplers and extra scheduler sigmas."""

        output = self.service_class().sample(
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
        )
        return (output,)
