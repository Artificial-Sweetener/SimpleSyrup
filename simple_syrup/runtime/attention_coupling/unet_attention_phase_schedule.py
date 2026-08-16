# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve standard-UNet attention composition across denoising."""

from __future__ import annotations

import math

import torch

from ..denoising_progress import DENOISING_PROGRESS_RESOLVER
from ..regional_attention_model_call_values import uniform_model_call_sigma
from .unet_attention_phase import (
    StandardUnetAttentionPhase,
    StandardUnetAttentionStage,
)


class StandardUnetAttentionPhaseSchedule:
    """Coordinate shared layout, regional ownership, and final cohesion."""

    def __init__(
        self,
        *,
        composition_fraction: float = 0.1,
        specialization_fraction: float = 0.55,
    ) -> None:
        """Set validated stage boundaries for one denoising trajectory."""

        values = (composition_fraction, specialization_fraction)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for value in values
        ):
            raise TypeError(
                "Standard UNet attention schedule values must be finite numbers."
            )
        self._composition_end = float(composition_fraction)
        self._specialization_end = self._composition_end + float(
            specialization_fraction
        )
        if not 0.0 < self._composition_end < self._specialization_end < 1.0:
            raise ValueError(
                "Standard UNet attention stages must fit within denoising."
            )

    def resolve(
        self,
        transformer_options: dict[str, object],
    ) -> StandardUnetAttentionPhase:
        """Resolve one attention phase from exact Comfy sampling metadata."""

        if not isinstance(transformer_options, dict):
            raise TypeError("Standard UNet phase options must be a dictionary.")
        sample_sigmas = transformer_options.get("sample_sigmas")
        current_sigmas = transformer_options.get("sigmas")
        if not isinstance(sample_sigmas, torch.Tensor):
            raise TypeError("Standard UNet phase requires tensor sample_sigmas.")
        if not isinstance(current_sigmas, torch.Tensor):
            raise TypeError("Standard UNet phase requires tensor sigmas.")
        progress = DENOISING_PROGRESS_RESOLVER.resolve(
            sample_sigmas,
            uniform_model_call_sigma(current_sigmas),
        )
        if progress < self._composition_end:
            return StandardUnetAttentionPhase(
                StandardUnetAttentionStage.COMPOSITION,
                progress,
                progress / self._composition_end,
            )
        if progress < self._specialization_end:
            return StandardUnetAttentionPhase(
                StandardUnetAttentionStage.SPECIALIZATION,
                progress,
                (progress - self._composition_end)
                / (self._specialization_end - self._composition_end),
            )
        return StandardUnetAttentionPhase(
            StandardUnetAttentionStage.CONSOLIDATION,
            progress,
            (progress - self._specialization_end) / (1.0 - self._specialization_end),
        )


STANDARD_UNET_ATTENTION_PHASE_SCHEDULE = StandardUnetAttentionPhaseSchedule()
