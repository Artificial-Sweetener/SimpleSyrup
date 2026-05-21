# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for KSampler Extras."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ..runtime import sampling_samplers, sampling_schedulers
from . import tooltips

Latent = dict[str, Any]


class KSamplerExtras:
    """Expose KSampler-style sampling with AYS and GITS scheduler options."""

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = (tooltips.DENOISED_LATENT_OUTPUT,)
    FUNCTION = "sample"
    CATEGORY = "SimpleSyrup/Sampling"
    DESCRIPTION = (
        "Denoises latents with extended sampler and scheduler options for "
        "compatible workflows."
    )
    SEARCH_ALIASES = ["ksampler", "sampler", "ays", "gits", "lcm"]

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
                    "CONDITIONING",
                    {"tooltip": tooltips.POSITIVE_CONDITIONING},
                ),
                "negative": (
                    "CONDITIONING",
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

        sampler = sampling_samplers.resolve_sampler(sampler_name)
        sigmas = sampling_schedulers.calculate_sigmas(
            model=model,
            scheduler_name=scheduler,
            sampler_name=sampler_name,
            steps=steps,
            denoise=denoise,
        ).to(model.load_device)

        latent_samples = latent_image["samples"]
        comfy_sample = _comfy_sample()
        comfy_utils = _comfy_utils()
        latent_preview = _latent_preview()

        latent_samples = comfy_sample.fix_empty_latent_channels(
            model,
            latent_samples,
            latent_image.get("downscale_ratio_spacial", None),
        )

        batch_inds = (
            latent_image["batch_index"] if "batch_index" in latent_image else None
        )
        noise = comfy_sample.prepare_noise(latent_samples, seed, batch_inds)
        noise_mask = latent_image.get("noise_mask", None)

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy_utils.PROGRESS_BAR_ENABLED
        samples = comfy_sample.sample_custom(
            model,
            noise,
            cfg,
            sampler,
            sigmas,
            positive,
            negative,
            latent_samples,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )

        output = latent_image.copy()
        output.pop("downscale_ratio_spacial", None)
        output["samples"] = samples
        return (output,)


def _comfy_sample() -> Any:
    """Import ComfyUI sample helpers lazily."""

    import comfy.sample

    return comfy.sample


def _comfy_utils() -> Any:
    """Import ComfyUI utility state lazily."""

    import comfy.utils

    return comfy.utils


def _latent_preview() -> Any:
    """Import ComfyUI preview helpers lazily."""

    return import_module("latent_preview")
