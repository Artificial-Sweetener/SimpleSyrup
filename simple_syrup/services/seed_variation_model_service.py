# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare a sampling MODEL with deterministic initial-noise variation."""

from __future__ import annotations

from ..domain.seed_variation import SeedVariationSettings
from ..runtime.seed_variation import SEED_VARIATION_MODEL_PATCH_BACKEND
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)


class SeedVariationModelService:
    """Validate seed variation and delegate MODEL derivation to the runtime backend."""

    def prepare(
        self,
        *,
        model: object,
        variation_seed: int,
        variation_strength: float,
    ) -> object:
        """Return a MODEL that varies the sampler's initial noise when enabled."""

        settings = SeedVariationSettings(
            variation_seed=variation_seed,
            strength=variation_strength,
        )
        derived = SEED_VARIATION_MODEL_PATCH_BACKEND.derive(model, settings)
        LOGGER.info(
            "Seed variation MODEL prepared",
            extra={
                "operation": "seed_variation_model_patch",
                "variation_seed": variation_seed,
                "variation_strength": settings.strength,
                "enabled": settings.strength > 0.0,
            },
        )
        return derived


SEED_VARIATION_MODEL_SERVICE = SeedVariationModelService()
