# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Measure opt-in denoising time for persistent standard-UNet variants."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrappersMP

from ..model_patcher_mutations import ModelKeyedWrapperMutation
from .standard_unet_cold_diagnostics import (
    STANDARD_UNET_COLD_PATH_DIAGNOSTICS,
    StandardUnetColdStage,
)

_WRAPPER_KEY = "simple_syrup.standard_unet_cold_sampling"


@dataclass(frozen=True, slots=True)
class StandardUnetColdSamplingDiagnosticsMutation:
    """Install one transparent sampler wrapper for opt-in cold attribution."""

    def apply(self, model: object) -> None:
        """Attach the keyed wrapper without changing ordinary sampling behavior."""

        if not isinstance(model, ModelPatcher):
            raise TypeError("Standard UNet cold sampling requires a MODEL.")
        ModelKeyedWrapperMutation(
            WrappersMP.SAMPLER_SAMPLE,
            _WRAPPER_KEY,
            self.measure_sampling,
        ).apply(model)

    def measure_sampling(
        self,
        executor: Callable[..., object],
        guider: object,
        sigmas: object,
        extra_args: object,
        callback: object,
        noise: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Measure one complete sampler call while forwarding exact arguments."""

        if not callable(executor):
            raise TypeError("Standard UNet cold sampling executor is invalid.")
        device = noise.device if isinstance(noise, torch.Tensor) else None
        with STANDARD_UNET_COLD_PATH_DIAGNOSTICS.measure(
            StandardUnetColdStage.SAMPLING,
            device=device,
        ) as metadata:
            result = executor(
                guider,
                sigmas,
                extra_args,
                callback,
                noise,
                *args,
                **kwargs,
            )
            if isinstance(sigmas, torch.Tensor):
                metadata["sigma_count"] = sigmas.numel()
            if isinstance(noise, torch.Tensor):
                metadata["latent_batch_size"] = int(noise.shape[0])
        return result
