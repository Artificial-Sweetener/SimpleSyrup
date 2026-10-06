# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Safeguard native starting-state verification independently of sampler callbacks."""

import pytest
import torch

from simple_syrup.runtime.noise_reuse_start import NoiseReuseStartVerification


def _native(
    sigma: torch.Tensor,
    noise: torch.Tensor,
    latent: torch.Tensor,
    maximum: bool,
) -> torch.Tensor:
    """Expose a native flow scaling boundary without sampler-specific machinery."""
    del maximum
    return sigma * noise + (1 - sigma) * latent


def _verification(expected: torch.Tensor) -> NoiseReuseStartVerification:
    """Attach recognizable workflow and selected-algorithm identity."""
    return NoiseReuseStartVerification(
        _native,
        expected,
        graph_context="Noise Reuse Options node 'Upscale' (ID 8) reaches Source (2).",
        sampler_name="uni_pc",
    )


def test_native_result_is_unchanged_and_only_initial_scaling_is_checked() -> None:
    """Permit algorithms to scale later states without rechecking the initial state."""
    latent, noise, sigma = (
        torch.ones(1, 4, 2, 2),
        torch.full((1, 4, 2, 2), 3.0),
        torch.tensor(0.4),
    )
    expected = _native(sigma, noise, latent, False)
    verification = _verification(expected)
    assert torch.equal(verification(sigma, noise, latent), expected)
    verification.require_verified()
    assert torch.equal(
        verification(sigma, noise + 1, latent), _native(sigma, noise + 1, latent, False)
    )


@pytest.mark.parametrize("failure", ("unused", "changed", "shape", "nonfinite"))
def test_failures_name_graph_node_and_selected_receiver(failure: str) -> None:
    """Name bypasses and double-noising independently of callback timing."""
    expected = torch.ones(1, 4, 2, 2)
    verification = _verification(expected)
    with pytest.raises(ValueError, match="Upscale.*ID 8.*uni_pc"):
        if failure == "unused":
            verification.require_verified()
        else:
            value = torch.zeros_like(expected)
            if failure == "shape":
                value = torch.zeros(1, 4, 1, 1)
            if failure == "nonfinite":
                value.fill_(float("nan"))
            verification(torch.tensor(0.4), value, value)
