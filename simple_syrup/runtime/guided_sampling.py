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
from .noise_reuse_sampling import prepare_reused_noise
from .noise_reuse_scope import current_noise_reuse
from .noise_reuse_start import NoiseReuseStartVerification

LOGGER = get_logger(__name__)


def sample_with_optional_negative(
    *,
    comfy_sample: Any,
    model: Any,
    noise: torch.Tensor,
    cfg: float,
    sampler: Any,
    sampler_name: str,
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

    reuse = current_noise_reuse()
    start_verification: NoiseReuseStartVerification | None = None
    if reuse is not None:
        if noise_inversion is not None:
            raise ValueError(
                "Noise reuse and full noise inversion cannot own the same "
                "sampling pass."
            )
        noise, expected = prepare_reused_noise(
            execution=reuse,
            model=model,
            fresh=noise,
            latent=latent_image,
            sigmas=sigmas,
            positive=positive,
            negative=negative,
            cfg=cfg,
            seed=seed,
            noise_mask=noise_mask,
            model_factory=inversion_model_factory,
        )
        start_verification = NoiseReuseStartVerification(
            model.get_model_object("model_sampling.noise_scaling"),
            expected,
            graph_context=reuse.provenance.graph_context,
            sampler_name=sampler_name,
        )
        model = start_verification.derive_model(model)

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
        samples = cast(
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
        if start_verification is not None:
            start_verification.require_verified()
        return samples

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
    if start_verification is not None:
        start_verification.require_verified()
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
