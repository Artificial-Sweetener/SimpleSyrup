# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Select ComfyUI's CFG or positive-only guider for latent sampling."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

import torch

from ..domain.noise_inversion import NoiseInversionOptions
from ..shared.logging import get_logger
from .noise_inversion import InversionModelFactory, invert_sampling_noise

LOGGER = get_logger(__name__)


def sample_with_optional_negative(
    *,
    comfy_sample: Any,
    model: Any,
    noise: torch.Tensor,
    cfg: float,
    sampler: Any,
    sigmas: torch.Tensor,
    positive: Any,
    negative: Any | None,
    latent_image: torch.Tensor,
    noise_mask: Any = None,
    callback: Any = None,
    disable_pbar: bool = False,
    seed: int | None = None,
    noise_inversion: NoiseInversionOptions | None = None,
    inversion_model_factory: InversionModelFactory | None = None,
) -> torch.Tensor:
    """Prepare optional source-derived noise and select the actual Comfy guider."""

    if noise_inversion is not None:
        inversion = invert_sampling_noise(
            model=model,
            latent=latent_image,
            forward_sigmas=sigmas,
            positive=positive,
            negative=negative,
            cfg=cfg,
            seed=seed,
            options=noise_inversion,
            model_factory=inversion_model_factory,
            noise_mask=noise_mask,
        )
        noise = inversion.noise.to(noise)

    if negative is not None:
        return cast(
            torch.Tensor,
            comfy_sample.sample_custom(
                model,
                noise,
                cfg,
                sampler,
                sigmas,
                positive,
                negative,
                latent_image,
                noise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed,
            ),
        )

    comfy_samplers = import_module("comfy.samplers")
    model_management = import_module("comfy.model_management")
    guider = comfy_samplers.CFGGuider(model)
    guider.inner_set_conds({"positive": positive})
    samples = guider.sample(
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    LOGGER.debug(
        "Positive-only ComfyUI guider selected",
        extra={
            "operation": "sample_with_optional_negative",
            "guidance_mode": "positive_only",
        },
    )
    return cast(
        torch.Tensor,
        samples.to(
            device=model_management.intermediate_device(),
            dtype=model_management.intermediate_dtype(),
        ),
    )
