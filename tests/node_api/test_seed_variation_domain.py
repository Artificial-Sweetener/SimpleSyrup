# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for validated seed-variation settings."""

from __future__ import annotations

import pytest

from simple_syrup.domain.seed_variation import (
    MAX_SEED,
    SeedVariationSettings,
)


def test_seed_variation_settings_normalize_valid_values() -> None:
    """Keep the full Comfy seed range and normalize numeric strength to float."""

    settings = SeedVariationSettings(variation_seed=MAX_SEED, strength=1)

    assert settings.variation_seed == MAX_SEED
    assert settings.strength == 1.0
    assert isinstance(settings.strength, float)


@pytest.mark.parametrize("variation_seed", [-1, MAX_SEED + 1])
def test_seed_variation_settings_reject_out_of_range_seeds(
    variation_seed: int,
) -> None:
    """Reject seeds outside ComfyUI's public unsigned 64-bit range."""

    with pytest.raises(ValueError, match="Variation seed must be between"):
        SeedVariationSettings(variation_seed=variation_seed, strength=0.5)


@pytest.mark.parametrize("variation_seed", [True, 1.5, "1"])
def test_seed_variation_settings_reject_non_integer_seeds(
    variation_seed: object,
) -> None:
    """Reject boolean and coercible values at the domain boundary."""

    with pytest.raises(TypeError, match="Variation seed must be an integer"):
        SeedVariationSettings(variation_seed=variation_seed, strength=0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("strength", [-0.01, 1.01, float("inf"), float("nan")])
def test_seed_variation_settings_reject_invalid_strength(strength: float) -> None:
    """Reject non-finite and out-of-range interpolation strengths."""

    with pytest.raises(ValueError, match="Variation strength must"):
        SeedVariationSettings(variation_seed=1, strength=strength)


@pytest.mark.parametrize("strength", [True, "0.5", None])
def test_seed_variation_settings_reject_non_numeric_strength(
    strength: object,
) -> None:
    """Reject values that would require implicit numeric coercion."""

    with pytest.raises(TypeError, match="Variation strength must be a number"):
        SeedVariationSettings(variation_seed=1, strength=strength)  # type: ignore[arg-type]
