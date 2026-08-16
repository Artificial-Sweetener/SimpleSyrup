# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Emit structured standard-UNet attention-phase diagnostics."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ...shared.logging import get_logger
from .unet_attention_phase import StandardUnetAttentionPhase

LOGGER = get_logger("runtime.attention_coupling.unet_attention_phase")


@dataclass(frozen=True, slots=True)
class StandardUnetAttentionPhaseDiagnostic:
    """Describe one exact model-call attention phase."""

    phase: StandardUnetAttentionPhase

    def __post_init__(self) -> None:
        """Require the authoritative phase value."""

        if not isinstance(self.phase, StandardUnetAttentionPhase):
            raise TypeError("Standard UNet attention phase diagnostic is invalid.")

    def to_log_fields(self) -> dict[str, object]:
        """Return JSON-safe phase fields for managed evidence."""

        return {
            "stage": self.phase.stage.value,
            "denoising_progress": self.phase.denoising_progress,
            "stage_progress": self.phase.stage_progress,
        }


class StandardUnetAttentionPhaseDiagnosticsEmitter:
    """Publish one structured phase record for every model call."""

    def __init__(self, logger: logging.Logger = LOGGER) -> None:
        """Retain the focused phase diagnostics logging boundary."""

        if not isinstance(logger, logging.Logger):
            raise TypeError("Standard UNet attention phase logger is invalid.")
        self._logger = logger

    def emit(self, diagnostic: StandardUnetAttentionPhaseDiagnostic) -> None:
        """Emit the phase without model inputs or tensor values."""

        if not isinstance(diagnostic, StandardUnetAttentionPhaseDiagnostic):
            raise TypeError("Standard UNet attention phase diagnostic is invalid.")
        if not self._logger.isEnabledFor(logging.DEBUG):
            return
        self._logger.debug(
            "Standard UNet attention phase",
            extra={
                "operation": "unet_attention_coupling.phase",
                "attention_phase_diagnostics": diagnostic.to_log_fields(),
            },
        )


STANDARD_UNET_ATTENTION_PHASE_DIAGNOSTICS_EMITTER = (
    StandardUnetAttentionPhaseDiagnosticsEmitter()
)
