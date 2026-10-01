# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026 Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define validated source-preserving noise inversion configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

InversionMethod = Literal["euler", "heun"]
INVERSION_METHODS: tuple[InversionMethod, ...] = ("euler", "heun")


@dataclass(frozen=True, slots=True)
class NoiseInversionOptions:
    """Configure reduced-resolution inversion and an optional full-size finish.

    The transition is a fraction of the target inversion sigma, not the forward
    denoise steps. A full-size inversion uses ``steps`` and needs no transfer.
    """

    method: InversionMethod = "euler"
    resolution_scale: float = 0.5
    steps: int = 2
    switch_fraction: float = 0.75
    finishing_steps: int = 1
    finishing_method: InversionMethod = "euler"

    def __post_init__(self) -> None:
        """Reject invalid or internally incomplete inversion recipes."""
        if self.method not in INVERSION_METHODS:
            raise ValueError("Inversion method must be euler or heun.")
        if self.finishing_method not in INVERSION_METHODS:
            raise ValueError("Inversion finishing method must be euler or heun.")
        if (
            not math.isfinite(self.resolution_scale)
            or not 0 < self.resolution_scale <= 1
        ):
            raise ValueError("Inversion resolution scale must be in (0, 1].")
        if type(self.steps) is not int or not 1 <= self.steps <= 64:
            raise ValueError("Inversion steps must be an integer between 1 and 64.")
        if type(self.finishing_steps) is not int or not 0 <= self.finishing_steps <= 64:
            raise ValueError("Inversion finishing steps must be between 0 and 64.")
        if not math.isfinite(self.switch_fraction) or not 0 < self.switch_fraction <= 1:
            raise ValueError("Inversion transition must be in (0, 1].")
        if self.resolution_scale < 1 and self.finishing_steps > 0:
            if self.switch_fraction == 1:
                raise ValueError("A full-size finish requires a transition below 100%.")

    @property
    def coarse_target_fraction(self) -> float:
        """Reach the full target unless an enabled full-size stage follows transfer."""
        return (
            self.switch_fraction
            if self.resolution_scale < 1 and self.finishing_steps
            else 1.0
        )

    def coarse_shape(self, height: int, width: int) -> tuple[int, int]:
        """Preserve full dimensions or align reduced transformer grids to even sizes."""
        if height < 1 or width < 1:
            raise ValueError("Inversion source dimensions must be positive.")
        if self.resolution_scale == 1:
            return height, width
        return (
            max(2, round(height * self.resolution_scale / 2) * 2),
            max(2, round(width * self.resolution_scale / 2) * 2),
        )
