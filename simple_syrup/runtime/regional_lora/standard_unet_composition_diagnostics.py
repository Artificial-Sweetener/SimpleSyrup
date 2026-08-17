# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Emit structured standard-UNet composition phase diagnostics."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from ...domain.regional_lora_plan import RegionalLoraScheduleBoundary
from ..attention_coupling.unet_attention_phase import StandardUnetAttentionPhase

_LOGGER = logging.getLogger(
    "simple_syrup.runtime.regional_lora.standard_unet_composition"
)


@dataclass(frozen=True, slots=True)
class StandardUnetCompositionDiagnostic:
    """Describe one exact model-call phase and authored schedule strengths."""

    attention_phase: StandardUnetAttentionPhase
    schedule_multipliers: tuple[float, ...]
    sampling_sigma: float
    adapter_schedules: tuple[tuple[RegionalLoraScheduleBoundary, ...], ...]
    spatial_modes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require one schedule and finite multiplier per canonical adapter."""

        if not math.isfinite(self.sampling_sigma):
            raise ValueError("Standard UNet composition sigma must be finite.")
        if len(self.schedule_multipliers) != len(self.adapter_schedules):
            raise ValueError(
                "Standard UNet composition schedules must align with multipliers."
            )
        if any(not schedule for schedule in self.adapter_schedules):
            raise ValueError("Standard UNet composition schedules must be nonempty.")
        if not self.spatial_modes or any(not mode for mode in self.spatial_modes):
            raise ValueError(
                "Standard UNet composition spatial modes must be nonempty."
            )

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible structured diagnostic values."""

        return {
            "stage": self.attention_phase.stage.value,
            "denoising_progress": self.attention_phase.denoising_progress,
            "sampling_sigma": self.sampling_sigma,
            "schedule_multipliers": list(self.schedule_multipliers),
            "adapter_schedules": [
                [
                    {
                        "start_percent": boundary.start_percent,
                        "start_sigma": boundary.start_sigma,
                        "strength_multiplier": boundary.strength_multiplier,
                        "guarantee_steps": boundary.guarantee_steps,
                    }
                    for boundary in schedule
                ]
                for schedule in self.adapter_schedules
            ],
            "spatial_modes": list(self.spatial_modes),
        }


class StandardUnetCompositionDiagnosticsEmitter:
    """Log one structured diagnostic for every regional model call."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Retain the configured logger without changing its level or handlers."""

        if logger is not None and not isinstance(logger, logging.Logger):
            raise TypeError("Standard UNet composition logger is invalid.")
        self._logger = logger or _LOGGER

    def emit(self, diagnostic: StandardUnetCompositionDiagnostic) -> None:
        """Emit one debug record with workflow-independent phase context."""

        if not isinstance(diagnostic, StandardUnetCompositionDiagnostic):
            raise TypeError("Standard UNet composition diagnostic is invalid.")
        self._logger.debug(
            "Resolved standard UNet regional composition phase",
            extra={
                "operation": "standard_unet_regional_composition.resolve",
                "regional_composition": diagnostic.as_dict(),
            },
        )


STANDARD_UNET_COMPOSITION_DIAGNOSTICS_EMITTER = (
    StandardUnetCompositionDiagnosticsEmitter()
)
