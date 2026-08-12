# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt one Anima composition phase to JSON-safe diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from .anima_composition_phase import AnimaCompositionPhase


@dataclass(frozen=True, slots=True)
class AnimaCompositionPhaseDiagnostics:
    """Expose the coordinated scene-building policy for one model call."""

    stage: str
    denoising_progress: float
    restrict_self_attention: bool
    regional_lora_scale: float

    @classmethod
    def from_phase(
        cls,
        phase: AnimaCompositionPhase,
    ) -> AnimaCompositionPhaseDiagnostics:
        """Capture one validated phase without retaining runtime state."""

        if not isinstance(phase, AnimaCompositionPhase):
            raise TypeError("Anima phase diagnostics require a composition phase.")
        return cls(
            stage=phase.stage.value,
            denoising_progress=phase.denoising_progress,
            restrict_self_attention=phase.restrict_self_attention,
            regional_lora_scale=phase.regional_lora_scale,
        )

    def to_log_fields(self) -> dict[str, object]:
        """Return the exact JSON-safe phase facts for structured logging."""

        return {
            "stage": self.stage,
            "denoising_progress": self.denoising_progress,
            "restrict_self_attention": self.restrict_self_attention,
            "regional_lora_scale": self.regional_lora_scale,
        }
