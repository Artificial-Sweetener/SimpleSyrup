# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for KSampler Extras."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from ..domain.conditioning_batch import ConditioningBatch, select_conditioning
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
        if _uses_conditioning_batch(positive, negative):
            samples = _sample_conditioning_batch(
                comfy_sample=comfy_sample,
                model=model,
                noise=noise,
                cfg=cfg,
                sampler=sampler,
                sigmas=sigmas,
                positive=positive,
                negative=negative,
                latent_samples=latent_samples,
                noise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed,
            )
        else:
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


def _uses_conditioning_batch(positive: Any, negative: Any) -> bool:
    """Return whether either conditioning input needs per-item selection."""

    return isinstance(positive, ConditioningBatch) or isinstance(
        negative,
        ConditioningBatch,
    )


def _sample_conditioning_batch(
    *,
    comfy_sample: Any,
    model: Any,
    noise: torch.Tensor,
    cfg: float,
    sampler: Any,
    sigmas: torch.Tensor,
    positive: Any,
    negative: Any,
    latent_samples: torch.Tensor,
    noise_mask: Any,
    callback: Any,
    disable_pbar: bool,
    seed: int,
) -> torch.Tensor:
    """Sample each latent batch item with its selected conditioning."""

    sampled: list[torch.Tensor] = []
    for index in range(int(latent_samples.shape[0])):
        sampled.append(
            comfy_sample.sample_custom(
                model,
                noise[index : index + 1],
                cfg,
                sampler,
                sigmas,
                select_conditioning(positive, index),
                select_conditioning(negative, index),
                latent_samples[index : index + 1],
                noise_mask=_slice_noise_mask(noise_mask, index, latent_samples),
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed,
            )
        )
    return torch.cat(sampled, dim=0)


def _slice_noise_mask(
    noise_mask: Any,
    index: int,
    latent_samples: torch.Tensor,
) -> Any:
    """Return the noise mask slice matching one latent batch item."""

    if isinstance(noise_mask, torch.Tensor) and noise_mask.shape[0] == int(
        latent_samples.shape[0],
    ):
        return noise_mask[index : index + 1]
    return noise_mask


def _comfy_utils() -> Any:
    """Import ComfyUI utility state lazily."""

    import comfy.utils

    return comfy.utils


def _latent_preview() -> Any:
    """Import ComfyUI preview helpers lazily."""

    return import_module("latent_preview")
