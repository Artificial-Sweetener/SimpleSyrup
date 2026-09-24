# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the focused SDXL visual sampling controls."""

from __future__ import annotations

import pytest

from tools.sdxl_attention_coupling_integration.sampling_controls import (
    MAX_COMFY_SEED,
    validate_sdxl_visual_seed,
)


@pytest.mark.parametrize("seed", (0, 7_429_113_058, MAX_COMFY_SEED))
def test_visual_seed_accepts_comfy_sampler_range(seed: int) -> None:
    """Return every integer inside Comfy's declared seed range unchanged."""

    assert validate_sdxl_visual_seed(seed) == seed


@pytest.mark.parametrize("seed", (-1, MAX_COMFY_SEED + 1))
def test_visual_seed_rejects_values_outside_comfy_sampler_range(seed: int) -> None:
    """Reject integers Comfy's sampler schema cannot represent."""

    with pytest.raises(ValueError, match="must be between"):
        validate_sdxl_visual_seed(seed)


@pytest.mark.parametrize("seed", (True, 1.5, "1"))
def test_visual_seed_rejects_non_integer_values(seed: object) -> None:
    """Reject bool and dynamically supplied non-integer values explicitly."""

    with pytest.raises(TypeError, match="must be an integer"):
        validate_sdxl_visual_seed(seed)
