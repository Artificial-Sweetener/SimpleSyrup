# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application service for ordinary KSampler-style latent sampling."""

from __future__ import annotations

from importlib import import_module
from typing import Any, TypeAlias

import torch

from ..domain.conditioning_batch import ConditioningBatch, select_conditioning
from ..runtime import sampling_samplers, sampling_schedulers
from ..shared.logging import get_logger

Latent: TypeAlias = dict[str, Any]
LOGGER = get_logger(__name__)


class KSamplerSamplingService:
    """Own reusable full-latent sampling while preserving batch semantics."""

    def sample(
        self,
        *,
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
    ) -> Latent:
        """Sample a latent with configured SimpleSyrup sampler extensions."""

        sampler = sampling_samplers.resolve_sampler(sampler_name)
        latent_samples = latent_image["samples"]
        if not isinstance(latent_samples, torch.Tensor):
            raise TypeError("KSampler latent samples must be a torch.Tensor.")

        comfy_sample = import_module("comfy.sample")
        comfy_utils = import_module("comfy.utils")
        latent_samples = comfy_sample.fix_empty_latent_channels(
            model,
            latent_samples,
            latent_image.get("downscale_ratio_spacial", None),
        )
        if not isinstance(latent_samples, torch.Tensor):
            raise TypeError(
                "KSampler normalized latent samples must be a torch.Tensor."
            )
        sigmas = sampling_schedulers.calculate_sigmas(
            model=model,
            scheduler_name=scheduler,
            sampler_name=sampler_name,
            steps=steps,
            denoise=denoise,
            view=sampling_schedulers.SchedulerView.from_tensor(latent_samples),
        ).to(model.load_device)
        noise = comfy_sample.prepare_noise(
            latent_samples,
            seed,
            latent_image.get("batch_index"),
        )
        noise_mask = latent_image.get("noise_mask")
        callback = import_module("latent_preview").prepare_callback(model, steps)
        disable_pbar = not comfy_utils.PROGRESS_BAR_ENABLED
        if self._uses_conditioning_batch(positive, negative):
            samples = self._sample_conditioning_batch(
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
        if not isinstance(samples, torch.Tensor):
            raise TypeError("KSampler output samples must be a torch.Tensor.")

        output = latent_image.copy()
        output.pop("downscale_ratio_spacial", None)
        output["samples"] = samples
        LOGGER.info(
            "KSampler pass completed",
            extra={
                "operation": "ksampler_sample",
                "sampler": sampler_name,
                "scheduler": scheduler,
                "steps": steps,
                "denoise": denoise,
                "latent_batch_size": int(samples.shape[0]),
                "latent_height": int(samples.shape[-2]),
                "latent_width": int(samples.shape[-1]),
            },
        )
        return output

    def _sample_conditioning_batch(
        self,
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
        """Sample latent items with existing per-item batch selection."""

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
                    noise_mask=self._slice_noise_mask(
                        noise_mask,
                        index,
                        latent_samples,
                    ),
                    callback=callback,
                    disable_pbar=disable_pbar,
                    seed=seed,
                )
            )
        return torch.cat(sampled, dim=0)

    def _slice_noise_mask(
        self,
        noise_mask: Any,
        index: int,
        latent_samples: torch.Tensor,
    ) -> Any:
        """Return a per-item noise mask when its batch matches the latent."""

        if isinstance(noise_mask, torch.Tensor) and noise_mask.shape[0] == int(
            latent_samples.shape[0]
        ):
            return noise_mask[index : index + 1]
        return noise_mask

    def _uses_conditioning_batch(self, positive: Any, negative: Any) -> bool:
        """Return whether existing per-latent conditioning selection applies."""

        return isinstance(positive, ConditioningBatch) or isinstance(
            negative, ConditioningBatch
        )
