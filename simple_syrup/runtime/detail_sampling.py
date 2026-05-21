# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI VAE and sampling adapters for scale-factor detailing."""

from __future__ import annotations

from importlib import import_module
from typing import Any, TypeAlias, cast

import torch

from . import sampling_samplers, sampling_schedulers
from .detail_previews import DetailPreviewContext, prepare_detail_preview_callback

Latent: TypeAlias = dict[str, Any]


class DetailSampler:
    """Adapt ComfyUI VAE and sampler APIs behind a testable boundary."""

    def encode(self, vae: Any, pixels: torch.Tensor, tiled: bool) -> Latent:
        """Encode pixels into a ComfyUI latent dictionary."""

        if tiled:
            nodes = _nodes()
            return cast(
                Latent,
                nodes.VAEEncodeTiled().encode(vae, pixels, 512, 64)[0],
            )
        return cast(Latent, _nodes().VAEEncode().encode(vae, pixels)[0])

    def decode(self, vae: Any, latent: Latent, tiled: bool) -> torch.Tensor:
        """Decode a ComfyUI latent dictionary into pixels."""

        if tiled:
            nodes = _nodes()
            return cast(
                torch.Tensor,
                nodes.VAEDecodeTiled().decode(vae, latent, 512, 64)[0],
            )
        return cast(torch.Tensor, _nodes().VAEDecode().decode(vae, latent)[0])

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
        denoise: float,
        preview_context: DetailPreviewContext | None = None,
    ) -> Latent:
        """Sample a latent with SimpleSyrup's sampler and scheduler helpers."""

        sampler = sampling_samplers.resolve_sampler(sampler_name)
        sigmas = sampling_schedulers.calculate_sigmas(
            model=model,
            scheduler_name=scheduler,
            sampler_name=sampler_name,
            steps=steps,
            denoise=denoise,
        ).to(model.load_device)

        latent_samples = cast(torch.Tensor, latent_image["samples"])
        comfy_sample = _comfy_sample()
        comfy_utils = _comfy_utils()

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
        if preview_context is None:
            callback = _latent_preview().prepare_callback(model, steps)
        else:
            callback = prepare_detail_preview_callback(model, steps, preview_context)
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
            disable_pbar=not comfy_utils.PROGRESS_BAR_ENABLED,
            seed=seed,
        )

        output = latent_image.copy()
        output.pop("downscale_ratio_spacial", None)
        output["samples"] = samples
        return output

    def apply_differential_diffusion(self, model: Any) -> Any:
        """Patch a model for feathered denoise masks when ComfyUI supports it."""

        options = getattr(model, "model_options", {})
        if (
            isinstance(options, dict)
            and options.get("denoise_mask_function") is not None
        ):
            return model

        module = import_module("comfy_extras.nodes_differential_diffusion")
        node = module.DifferentialDiffusion
        output = node.execute(model, 1.0)
        if hasattr(output, "result"):
            return output.result[0]
        if isinstance(output, tuple):
            return output[0]
        return output[0]


def _nodes() -> Any:
    """Import ComfyUI core nodes lazily."""

    return import_module("nodes")


def _comfy_sample() -> Any:
    """Import ComfyUI sampling helpers lazily."""

    import comfy.sample

    return comfy.sample


def _comfy_utils() -> Any:
    """Import ComfyUI utility state lazily."""

    import comfy.utils

    return comfy.utils


def _latent_preview() -> Any:
    """Import ComfyUI preview helpers lazily."""

    return import_module("latent_preview")
