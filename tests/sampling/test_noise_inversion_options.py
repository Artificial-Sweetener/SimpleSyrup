# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Safeguard the accepted inversion recipe and independently editable controls."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from simple_syrup.domain.noise_inversion import NoiseInversionOptions


def test_defaults_match_accepted_half_euler_recipe() -> None:
    """Retain the user-accepted reduced-size path and full-size finishing stage."""
    options = NoiseInversionOptions()
    assert options.method == "euler"
    assert options.resolution_scale == 0.5
    assert options.steps == 2
    assert options.switch_fraction == 0.75
    assert options.finishing_steps == 1
    assert options.finishing_method == "euler"
    assert options.coarse_target_fraction == 0.75


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"method": "fireflow"}, "method"),
        ({"finishing_method": "invalid"}, "finishing method"),
        ({"resolution_scale": 0}, "resolution"),
        ({"resolution_scale": 1.1}, "resolution"),
        ({"resolution_scale": float("nan")}, "resolution"),
        ({"steps": 0}, "steps"),
        ({"steps": 65}, "steps"),
        ({"steps": True}, "steps"),
        ({"steps": 1.5}, "steps"),
        ({"finishing_steps": -1}, "finishing steps"),
        ({"finishing_steps": 65}, "finishing steps"),
        ({"switch_fraction": 0}, "transition"),
        ({"switch_fraction": float("inf")}, "transition"),
        ({"switch_fraction": 1}, "finish"),
    ],
)
def test_invalid_configuration_is_rejected(
    changes: dict[str, Any], message: str
) -> None:
    """Fail before any inversion/model execution for invalid graph controls."""
    with pytest.raises(ValueError, match=message):
        NoiseInversionOptions(**changes)


def test_no_finish_reaches_target_without_changing_the_saved_transition() -> None:
    """Make finishing-step zero a usable coarse-only configuration."""
    options = NoiseInversionOptions(finishing_steps=0)
    assert options.coarse_target_fraction == 1
    assert options.switch_fraction == 0.75


def test_full_resolution_preserves_odd_dimensions_and_uses_one_stage() -> None:
    """Never round an existing full-size latent onto a different grid."""
    options = NoiseInversionOptions(resolution_scale=1)
    assert options.coarse_shape(129, 203) == (129, 203)
    assert options.coarse_target_fraction == 1


@pytest.mark.parametrize(
    ("height", "width", "scale", "expected"),
    [(144, 252, 0.5, (72, 126)), (129, 203, 0.5, (64, 102)), (1, 1, 0.25, (2, 2))],
)
def test_reduced_resolution_matches_experimental_even_grid(
    height: int, width: int, scale: float, expected: tuple[int, int]
) -> None:
    """Keep reduced transformer-grid rounding consistent with the accepted runs."""
    assert (
        NoiseInversionOptions(resolution_scale=scale).coarse_shape(height, width)
        == expected
    )


def test_configuration_cannot_be_mutated() -> None:
    """Keep branched options chains isolated from downstream edits."""
    field_name = "steps"
    with pytest.raises(FrozenInstanceError):
        setattr(NoiseInversionOptions(), field_name, 8)


def test_invalid_source_shape_is_rejected() -> None:
    """Reject impossible source dimensions before spatial projection."""
    with pytest.raises(ValueError, match="positive"):
        NoiseInversionOptions().coarse_shape(0, 128)
