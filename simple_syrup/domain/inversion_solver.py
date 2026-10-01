# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026 Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Integrate source-derived inversion states without ComfyUI dependencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .noise_inversion import INVERSION_METHODS, InversionMethod

InversionVelocity = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
SpatialResize = Callable[[torch.Tensor, int, int], torch.Tensor]


@dataclass(slots=True)
class InversionSolverEvidence:
    """Count the actual denoiser evaluations performed by an integration stage."""

    evaluations: int = 0


def integrate_inversion(
    source: torch.Tensor,
    sigmas: torch.Tensor,
    evaluate: InversionVelocity,
    *,
    method: InversionMethod,
    evidence: InversionSolverEvidence | None = None,
) -> torch.Tensor:
    """Advance a finite state over strictly increasing positive inversion sigmas."""
    if method not in INVERSION_METHODS:
        raise ValueError("Inversion method must be euler or heun.")
    if sigmas.ndim != 1 or len(sigmas) < 2 or not bool(torch.isfinite(sigmas).all()):
        raise ValueError("A finite one-dimensional inversion schedule is required.")
    if not bool(torch.all(sigmas > 0)) or not bool(torch.all(sigmas[1:] > sigmas[:-1])):
        raise ValueError("Inversion sigmas must be positive and increasing.")
    if not source.is_floating_point() or not bool(torch.isfinite(source).all()):
        raise ValueError("Inversion source must contain finite floating-point values.")
    record = evidence if evidence is not None else InversionSolverEvidence()
    state = source.clone()

    def velocity(x: torch.Tensor, sigma: torch.Tensor, index: int) -> torch.Tensor:
        """Count every denoiser evaluation and reject corrupted predictions."""
        record.evaluations += 1
        value = evaluate(x, sigma, index)
        if value.shape != x.shape or not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Invalid inversion velocity shape or values.")
        return value

    for index in range(len(sigmas) - 1):
        current, following = sigmas[index], sigmas[index + 1]
        delta = following - current
        estimate = velocity(state, current, index)
        if method == "heun":
            corrected = velocity(state + delta * estimate, following, index)
            estimate = (estimate + corrected) / 2
        state = state + delta * estimate
        if not bool(torch.isfinite(state).all()):
            raise FloatingPointError(f"Non-finite inversion state at step {index}.")
    return state


def lift_inversion_displacement(
    full_source: torch.Tensor,
    coarse_source: torch.Tensor,
    coarse_endpoint: torch.Tensor,
    *,
    resize: SpatialResize,
) -> torch.Tensor:
    """Lift only the inferred change so existing full-size detail survives transfer."""
    if coarse_source.shape != coarse_endpoint.shape:
        raise ValueError("Coarse inversion source and endpoint shapes must match.")
    if full_source.shape[:-2] != coarse_source.shape[:-2]:
        raise ValueError(
            "Inversion transfer must preserve batch and channel dimensions."
        )
    height, width = full_source.shape[-2:]
    lifted = resize(coarse_endpoint - coarse_source, height, width)
    if lifted.shape != full_source.shape:
        raise ValueError("Inversion displacement resize produced an invalid shape.")
    endpoint = full_source + lifted
    if not bool(torch.isfinite(endpoint).all()):
        raise FloatingPointError("Inversion transfer produced non-finite values.")
    return endpoint
