# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file are adapted from AUTOMATIC1111 stable-diffusion-webui
# and k-diffusion. See third_party/manifest.toml and third_party/NOTICE.md.

"""AUTOMATIC1111-derived sampler functions for SimpleSyrup sampling nodes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from types import ModuleType
from typing import Any, Protocol, cast

import torch


class DenoiseModel(Protocol):
    """Represent the k-diffusion denoiser callable used by sampler functions."""

    def __call__(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Denoise a latent tensor at the requested sigma."""


NoiseSampler = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
SamplerCallback = Callable[[dict[str, object]], None]


@torch.no_grad()
def sample_euler_ancestral_a1111(
    model: DenoiseModel,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Mapping[str, object] | None = None,
    callback: SamplerCallback | None = None,
    disable: bool | None = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler: NoiseSampler | None = None,
) -> torch.Tensor:
    """Run A1111/k-diffusion Euler ancestral math with ComfyUI seed plumbing."""

    del disable
    normalized_extra_args = {} if extra_args is None else dict(extra_args)
    active_noise_sampler = noise_sampler or _default_noise_sampler(
        x,
        normalized_extra_args.get("seed"),
    )
    s_in = x.new_ones([x.shape[0]])

    for index in range(len(sigmas) - 1):
        denoised = model(x, sigmas[index] * s_in, **normalized_extra_args)
        sigma_down, sigma_up = _get_ancestral_step(
            sigmas[index],
            sigmas[index + 1],
            eta=eta,
        )
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": index,
                    "sigma": sigmas[index],
                    "sigma_hat": sigmas[index],
                    "denoised": denoised,
                }
            )

        derivative = _to_d(x, sigmas[index], denoised)
        dt = sigma_down - sigmas[index]
        x = x + derivative * dt
        if sigmas[index + 1] > 0:
            x = (
                x
                + active_noise_sampler(sigmas[index], sigmas[index + 1])
                * s_noise
                * sigma_up
            )
    return x


def _default_noise_sampler(x: torch.Tensor, seed: object) -> NoiseSampler:
    """Return ComfyUI's deterministic noise sampler for the active seed."""

    try:
        default_noise_sampler = cast(
            Any,
            _comfy_k_diffusion_sampling(),
        ).default_noise_sampler
    except AttributeError as error:
        raise ValueError("ComfyUI default_noise_sampler is unavailable.") from error
    if not callable(default_noise_sampler):
        raise ValueError("ComfyUI default_noise_sampler is unavailable.")
    return cast(NoiseSampler, default_noise_sampler(x, seed=seed))


def _get_ancestral_step(
    sigma_from: torch.Tensor,
    sigma_to: torch.Tensor,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the ancestral down-step and noise scale with ComfyUI helpers."""

    try:
        get_ancestral_step = cast(
            Any,
            _comfy_k_diffusion_sampling(),
        ).get_ancestral_step
    except AttributeError as error:
        raise ValueError("ComfyUI get_ancestral_step is unavailable.") from error
    if not callable(get_ancestral_step):
        raise ValueError("ComfyUI get_ancestral_step is unavailable.")
    return cast(
        tuple[torch.Tensor, torch.Tensor],
        get_ancestral_step(sigma_from, sigma_to, eta=eta),
    )


def _to_d(
    x: torch.Tensor,
    sigma: torch.Tensor,
    denoised: torch.Tensor,
) -> torch.Tensor:
    """Convert denoised output to an Euler derivative with ComfyUI helpers."""

    try:
        to_d = cast(Any, _comfy_k_diffusion_sampling()).to_d
    except AttributeError as error:
        raise ValueError("ComfyUI to_d is unavailable.") from error
    if not callable(to_d):
        raise ValueError("ComfyUI to_d is unavailable.")
    return cast(torch.Tensor, to_d(x, sigma, denoised))


def _comfy_k_diffusion_sampling() -> ModuleType:
    """Import ComfyUI's k-diffusion sampling helpers lazily."""

    return import_module("comfy.k_diffusion.sampling")
