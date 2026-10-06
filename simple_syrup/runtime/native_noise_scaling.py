# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Separate clean references and residuals with the model's actual parameterization."""

from __future__ import annotations

from typing import Any

import torch


def noise_coefficients(
    scaling: Any, clean: torch.Tensor, coordinate: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate affine noise mixing instead of assuming DDPM alpha/sigma units."""
    sigma = clean.new_tensor(coordinate)
    base = scaling.noise_scaling(sigma, torch.zeros_like(clean), clean, False)
    zero = torch.zeros_like(clean)
    amplitude = scaling.noise_scaling(sigma, torch.ones_like(clean), zero, False)
    amplitude = amplitude - scaling.noise_scaling(sigma, zero, zero, False)
    if not isinstance(base, torch.Tensor) or not isinstance(amplitude, torch.Tensor):
        raise TypeError("Native noise scaling must produce tensor coefficients.")
    if base.shape != clean.shape or amplitude.shape != clean.shape:
        raise ValueError("Native noise scaling must preserve the clean latent shape.")
    if not bool(torch.isfinite(amplitude).all()) or bool((amplitude == 0).any()):
        raise ValueError(
            "Capture coordinate must have finite invertible noise amplitude."
        )
    probe = scaling.noise_scaling(sigma, torch.full_like(clean, 0.37), clean, False)
    if not isinstance(probe, torch.Tensor) or not torch.allclose(
        probe, base + amplitude * 0.37, atol=2e-5, rtol=2e-5
    ):
        raise ValueError("Native noise scaling must be affine in its noise argument.")
    return base, amplitude


def recover_residual(
    scaling: Any, clean: torch.Tensor, state: torch.Tensor, coordinate: float
) -> torch.Tensor:
    """Require the recovered residual to reconstruct its actual captured state."""
    if clean.shape != state.shape:
        raise ValueError("Captured state and original clean reference shapes differ.")
    base, amplitude = noise_coefficients(scaling, clean, coordinate)
    residual = (state - base) / amplitude
    if not bool(torch.isfinite(residual).all()) or not torch.allclose(
        base + amplitude * residual, state, atol=2e-5, rtol=2e-5
    ):
        raise ValueError("Recovered residual cannot reconstruct the captured state.")
    return residual
