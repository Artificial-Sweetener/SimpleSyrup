# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define phased Anima composition, specialization, and consolidation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch


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

        values = self._validated_sigmas(sample_sigmas)
        progress = self._progress(values, self._finite_scalar(current_sigma))
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

    @staticmethod
    def _progress(values: tuple[float, ...], sigma: float) -> float:
        """Interpolate denoising progress between adjacent schedule sigmas."""

        steps = len(values) - 1
        if sigma >= values[0]:
            return 0.0
        if sigma <= values[-1]:
            return 1.0
        for index, (start, end) in enumerate(zip(values, values[1:], strict=True)):
            if start >= sigma >= end:
                span = start - end
                fraction = 0.0 if span == 0.0 else (start - sigma) / span
                return (index + fraction) / steps
        raise ValueError("Anima current sigma lies outside its sampling schedule.")

    @staticmethod
    def _validated_sigmas(sample_sigmas: torch.Tensor) -> tuple[float, ...]:
        """Return one finite descending schedule with a terminal sample value."""

        if (
            not isinstance(sample_sigmas, torch.Tensor)
            or not sample_sigmas.is_floating_point()
            or sample_sigmas.ndim != 1
            or sample_sigmas.numel() < 2
        ):
            raise ValueError(
                "Anima composition sample_sigmas must be a floating vector with "
                "at least two values."
            )
        values = tuple(float(value) for value in sample_sigmas.detach().cpu().tolist())
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Anima composition sample_sigmas must be finite.")
        if any(values[index] < values[index + 1] for index in range(len(values) - 1)):
            raise ValueError("Anima composition sample_sigmas must be descending.")
        return values

    @staticmethod
    def _finite_scalar(value: object) -> float:
        """Normalize one scalar tensor or real current sigma."""

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Anima composition current sigma must be scalar.")
            value = value.detach().cpu().item()
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("Anima composition current sigma must be a real number.")
        sigma = float(value)
        if not math.isfinite(sigma):
            raise ValueError("Anima composition current sigma must be finite.")
        return sigma


ANIMA_COMPOSITION_PHASE_SCHEDULE = AnimaCompositionPhaseSchedule()
