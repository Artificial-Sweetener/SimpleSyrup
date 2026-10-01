# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove inversion accuracy, source preservation, and finite-state safety."""

from __future__ import annotations

import math

import pytest
import torch

from simple_syrup.domain.inversion_solver import (
    InversionSolverEvidence,
    integrate_inversion,
    lift_inversion_displacement,
)
from simple_syrup.domain.noise_inversion import InversionMethod
from simple_syrup.runtime.spatial_tensor_projection import resize_spatial_tensor


@pytest.mark.parametrize("method", ["euler", "heun"])
def test_constant_velocity_has_exact_endpoint_and_observed_call_count(
    method: InversionMethod,
) -> None:
    """Match a closed-form flow path rather than duplicating integration logic."""
    source = torch.zeros((1, 2, 4, 4))
    evidence = InversionSolverEvidence()
    calls: list[tuple[float, int]] = []

    def velocity(x: torch.Tensor, sigma: torch.Tensor, index: int) -> torch.Tensor:
        """Record neural-evaluation positions and return a constant flow."""
        calls.append((float(sigma), index))
        return torch.full_like(x, 2)

    endpoint = integrate_inversion(
        source,
        torch.tensor([0.125, 0.25, 0.5]),
        velocity,
        method=method,
        evidence=evidence,
    )
    assert torch.equal(endpoint, torch.full_like(source, 0.75))
    assert torch.count_nonzero(source) == 0
    assert endpoint.data_ptr() != source.data_ptr()
    assert evidence.evaluations == len(calls) == (2 if method == "euler" else 4)


@pytest.mark.parametrize(("method", "expected"), [("euler", 1.5), ("heun", 1.625)])
def test_nonconstant_velocity_distinguishes_euler_and_heun(
    method: InversionMethod, expected: float
) -> None:
    """Prove Heun evaluates its predictor and averages the endpoint velocity."""
    endpoint = integrate_inversion(
        torch.ones((1, 1, 2, 2)),
        torch.tensor([0.25, 0.75]),
        lambda x, sigma, index: x,
        method=method,
    )
    assert torch.equal(endpoint, torch.full_like(endpoint, expected))


@pytest.mark.parametrize(
    "sigmas",
    [
        torch.tensor([0.0, 0.5]),
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.5, 0.25]),
        torch.tensor([0.1, float("nan")]),
        torch.tensor([0.1]),
        torch.ones((2, 2)),
    ],
)
def test_invalid_schedule_fails_before_evaluation(sigmas: torch.Tensor) -> None:
    """Reject malformed schedules without calling the model."""

    def forbidden(x: torch.Tensor, sigma: torch.Tensor, index: int) -> torch.Tensor:
        """Detect an invalid request reaching the expensive model boundary."""
        raise AssertionError("Invalid schedule reached evaluation.")

    with pytest.raises(ValueError, match="schedule|sigmas"):
        integrate_inversion(torch.ones((1, 1, 2, 2)), sigmas, forbidden, method="euler")


@pytest.mark.parametrize("wrong_shape", [False, True])
def test_bad_velocity_is_rejected(wrong_shape: bool) -> None:
    """Never continue integrating corrupted predictions."""

    def velocity(x: torch.Tensor, sigma: torch.Tensor, index: int) -> torch.Tensor:
        """Produce one malformed prediction at the runtime boundary."""
        return torch.zeros((1,)) if wrong_shape else torch.full_like(x, float("inf"))

    with pytest.raises(FloatingPointError, match="velocity"):
        integrate_inversion(
            torch.ones((1, 1, 2, 2)), torch.tensor([0.1, 0.5]), velocity, method="euler"
        )


def _resize(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Use the production spatial projector for displacement transfer."""
    return resize_spatial_tensor(tensor, height=height, width=width, mode="bilinear")


def test_displacement_transfer_preserves_full_source_high_frequency_detail() -> None:
    """Keep original detail instead of enlarging a blurred low-resolution endpoint."""
    full_source = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) % 2
    coarse_source = torch.full((1, 1, 4, 4), 0.5)
    endpoint = lift_inversion_displacement(
        full_source, coarse_source, coarse_source + 2, resize=_resize
    )
    assert torch.equal(endpoint, full_source + 2)


@pytest.mark.parametrize("singleton_depth", [False, True])
def test_zero_displacement_is_pixel_exact(singleton_depth: bool) -> None:
    """Preserve both supported latent layouts without losing source pixels."""
    shape = (2, 4, 1, 7, 9) if singleton_depth else (2, 4, 7, 9)
    source = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    coarse = resize_spatial_tensor(source, height=4, width=4, mode="area")
    assert torch.equal(
        lift_inversion_displacement(source, coarse, coarse, resize=_resize), source
    )


def test_transfer_rejects_mismatched_sources() -> None:
    """Reject a transfer that would broadcast across different latent batches."""
    with pytest.raises(ValueError, match="batch"):
        lift_inversion_displacement(
            torch.zeros((2, 1, 8, 8)),
            torch.zeros((1, 1, 4, 4)),
            torch.zeros((1, 1, 4, 4)),
            resize=_resize,
        )
