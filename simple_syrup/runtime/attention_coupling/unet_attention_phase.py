# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Describe one standard-UNet attention-composition phase."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class StandardUnetAttentionStage(Enum):
    """Identify the active standard-UNet attention responsibility."""

    COMPOSITION = "composition"
    SPECIALIZATION = "specialization"
    CONSOLIDATION = "consolidation"


@dataclass(frozen=True, slots=True)
class StandardUnetAttentionPhase:
    """Publish one normalized attention stage and its local progress."""

    stage: StandardUnetAttentionStage
    denoising_progress: float
    stage_progress: float

    def __post_init__(self) -> None:
        """Require finite normalized phase values and explicit ownership."""

        if not isinstance(self.stage, StandardUnetAttentionStage):
            raise TypeError("Standard UNet attention stage is invalid.")
        for name, value in (
            ("denoising progress", self.denoising_progress),
            ("stage progress", self.stage_progress),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Standard UNet attention {name} must be in [0, 1].")
