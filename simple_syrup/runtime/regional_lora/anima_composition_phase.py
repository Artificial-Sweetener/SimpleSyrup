# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define phased Anima composition, specialization, and consolidation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch

from ..denoising_progress import DENOISING_PROGRESS_RESOLVER


class AnimaCompositionStage(Enum):
    """Identify the active denoising responsibility."""

    COMPOSITION = "composition"
    SPECIALIZATION = "specialization"
    CONSOLIDATION = "consolidation"


@dataclass(frozen=True, slots=True)
class AnimaCompositionPhase:
    """Publish coordinated self-attention and LoRA behavior."""

    stage: AnimaCompositionStage
    denoising_progress: float
    restrict_self_attention: bool
    regional_lora_scale: float

    def __post_init__(self) -> None:
        """Require finite normalized phase values."""

        if not isinstance(self.stage, AnimaCompositionStage):
            raise TypeError("Anima composition phase stage is invalid.")
        for name, value in (
            ("progress", self.denoising_progress),
            ("LoRA scale", self.regional_lora_scale),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Anima composition {name} must be in [0, 1].")


class AnimaCompositionPhaseSchedule:
    """Coordinate one-scene layout, regional identity, and final cohesion."""

    def __init__(
        self,
        *,
        composition_fraction: float = 0.1,
        specialization_fraction: float = 0.55,
        final_lora_scale: float = 0.85,
    ) -> None:
        """Set validated stage boundaries and final identity retention."""

        values = (
            composition_fraction,
            specialization_fraction,
            final_lora_scale,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for value in values
        ):
            raise TypeError("Anima composition schedule values must be finite numbers.")
        self._composition_end = float(composition_fraction)
        self._specialization_end = self._composition_end + float(
            specialization_fraction
        )
        self._final_lora_scale = float(final_lora_scale)
        if not 0.0 < self._composition_end < self._specialization_end < 1.0:
            raise ValueError("Anima composition stages must fit within denoising.")
        if not 0.0 <= self._final_lora_scale <= 1.0:
            raise ValueError("Anima final LoRA scale must be in [0, 1].")

    def resolve(
        self,
        sample_sigmas: torch.Tensor,
        current_sigma: object,
    ) -> AnimaCompositionPhase:
        """Resolve one coordinated phase from the exact Comfy sigma schedule."""

        progress = DENOISING_PROGRESS_RESOLVER.resolve(sample_sigmas, current_sigma)
        if progress < self._composition_end:
            local = progress / self._composition_end
            return AnimaCompositionPhase(
                AnimaCompositionStage.COMPOSITION,
                progress,
                False,
                0.25 * self._smoothstep(local),
            )
        if progress < self._specialization_end:
            local = (progress - self._composition_end) / (
                self._specialization_end - self._composition_end
            )
            lora_ramp = self._smoothstep(min(1.0, local / 0.3))
            return AnimaCompositionPhase(
                AnimaCompositionStage.SPECIALIZATION,
                progress,
                True,
                0.25 + (0.75 * lora_ramp),
            )
        local = (progress - self._specialization_end) / (1.0 - self._specialization_end)
        consolidation = self._smoothstep(local)
        return AnimaCompositionPhase(
            AnimaCompositionStage.CONSOLIDATION,
            progress,
            False,
            1.0 - ((1.0 - self._final_lora_scale) * consolidation),
        )

    @staticmethod
    def _smoothstep(value: float) -> float:
        """Return a continuous zero-slope interpolation over [0, 1]."""

        bounded = min(1.0, max(0.0, value))
        return bounded * bounded * (3.0 - (2.0 * bounded))


ANIMA_COMPOSITION_PHASE_SCHEDULE = AnimaCompositionPhaseSchedule()
