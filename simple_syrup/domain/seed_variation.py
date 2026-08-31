# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define validated seed-variation sampling settings."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

MIN_SEED = 0
MAX_SEED = 0xFFFFFFFFFFFFFFFF
MIN_VARIATION_STRENGTH = 0.0
MAX_VARIATION_STRENGTH = 1.0


@dataclass(frozen=True, slots=True)
class SeedVariationSettings:
    """Hold one deterministic initial-noise interpolation request."""

    variation_seed: int
    strength: float

    def __post_init__(self) -> None:
        """Reject settings outside ComfyUI's public seed and strength ranges."""

        if isinstance(self.variation_seed, bool) or not isinstance(
            self.variation_seed,
            int,
        ):
            raise TypeError("Variation seed must be an integer.")
        if not MIN_SEED <= self.variation_seed <= MAX_SEED:
            raise ValueError(
                f"Variation seed must be between {MIN_SEED} and {MAX_SEED}."
            )
        if isinstance(self.strength, bool) or not isinstance(
            self.strength,
            (int, float),
        ):
            raise TypeError("Variation strength must be a number.")
        normalized_strength = float(self.strength)
        if not isfinite(normalized_strength):
            raise ValueError("Variation strength must be finite.")
        if (
            not MIN_VARIATION_STRENGTH
            <= normalized_strength
            <= (MAX_VARIATION_STRENGTH)
        ):
            raise ValueError(
                "Variation strength must be between "
                f"{MIN_VARIATION_STRENGTH} and {MAX_VARIATION_STRENGTH}."
            )
        object.__setattr__(self, "strength", normalized_strength)
