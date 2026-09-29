# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own initial latent noise generation for core and RES4LYF samplers."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol, cast

import torch

from .res4lyf_sampler_names import RES4LYF_SAMPLER_NAMES


class SamplingModel(Protocol):
    """Expose the model sampling object used by RES4LYF noise generation."""

    def get_model_object(self, name: str) -> object:
        """Return a named ComfyUI model object."""


class ModelSamplingBounds(Protocol):
    """Expose the sigma limits used by RES4LYF's noise generator."""

    sigma_max: float | torch.Tensor
    sigma_min: float | torch.Tensor


def prepare_sampling_noise(
    *,
    comfy_sample: Any,
    sampler_name: str,
    samples: torch.Tensor,
    seed: int,
    batch_indices: Any,
    model: SamplingModel,
) -> torch.Tensor:
    """Generate RES4LYF's default noise or preserve ComfyUI's core path."""

    if sampler_name not in RES4LYF_SAMPLER_NAMES:
        return cast(
            torch.Tensor, comfy_sample.prepare_noise(samples, seed, batch_indices)
        )

    model_sampling = cast(ModelSamplingBounds, model.get_model_object("model_sampling"))
    sigma_max = model_sampling.sigma_max
    sigma_min = model_sampling.sigma_min
    noise_classes = import_module(
        "simple_syrup.third_party.res4lyf_runtime.beta.noise_classes"
    )
    latents = import_module("simple_syrup.third_party.res4lyf_runtime.latents")
    reference = samples.to(torch.float32)
    generator = noise_classes.NOISE_GENERATOR_CLASSES_SIMPLE["gaussian"](
        x=reference.to(torch.float64),
        seed=seed,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
    )
    noise = cast(torch.Tensor, generator(sigma=sigma_max, sigma_next=sigma_min))
    if noise.std() > 0:
        noise = cast(
            torch.Tensor,
            latents.normalize_zscore(noise, channelwise=True, inplace=True),
        )
    noise = noise - noise.mean()
    return noise.to(reference.dtype)
