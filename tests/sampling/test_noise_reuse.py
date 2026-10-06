# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify captured coefficient identity, coordinate selection and option isolation."""

import pytest
import torch

from simple_syrup.domain.noise_inversion import NoiseInversionOptions
from simple_syrup.domain.noise_reuse import (
    CapturedSamplingState,
    NoiseReuseOptions,
    SamplingProvenance,
)
from simple_syrup.domain.sampler_options import (
    SamplerOptions,
    append_sampler_capability,
)
from simple_syrup.domain.spectral_noise import lift_residual, spatial_dct


def test_lifting_keeps_source_coefficients_and_completes_only_new_bands() -> None:
    """Detect interpolation, whitening and replacement of captured coefficients."""
    generator = torch.Generator().manual_seed(17)
    source = torch.randn((1, 16, 1, 5, 7), generator=generator) * 0.7 + 0.2
    fresh = torch.randn((1, 16, 1, 13, 18), generator=generator)
    lifted = spatial_dct(lift_residual(source, fresh))
    assert torch.allclose(lifted[..., :5, :7], spatial_dct(source), atol=2e-6)
    difference = lifted - spatial_dct(fresh)
    difference[..., :5, :7] = 0
    assert float(difference.abs().max()) < 2e-6


def test_same_size_reuses_residual_and_wrong_layout_fails() -> None:
    """Avoid extra noise at unchanged size and reject silently broadcast batches."""
    source = torch.arange(24).float().reshape(1, 1, 4, 6)
    assert torch.allclose(
        lift_residual(source, torch.zeros_like(source)), source, atol=3e-6
    )
    with pytest.raises(ValueError, match="batch"):
        lift_residual(source, torch.zeros(2, 1, 8, 12))
    with pytest.raises(ValueError, match="enlargement"):
        lift_residual(source, torch.zeros(1, 1, 2, 3))


def test_finishing_selects_its_actual_start_and_zero_selects_sampler_start() -> None:
    """Keep sampler denoise separate from inversion finishing work."""
    target = float(torch.tensor(0.41665297746658325))
    assert NoiseReuseOptions().capture_coordinate(target) == target
    assert NoiseReuseOptions(finishing_steps=1).capture_coordinate(target) == float(
        torch.tensor(target * 0.75)
    )
    with pytest.raises(ValueError, match="fraction"):
        NoiseReuseOptions(finishing_start_fraction=0)
    with pytest.raises(ValueError, match="steps"):
        NoiseReuseOptions(finishing_steps=-1)


def test_options_branch_without_mutating_prefix_and_reject_full_inversion() -> None:
    """Reject two competing starting-noise owners in either connection order."""
    prefix = SamplerOptions()
    reuse = append_sampler_capability(prefix, NoiseReuseOptions())
    assert prefix.noise_reuse is None
    assert reuse.noise_reuse == NoiseReuseOptions()
    with pytest.raises(ValueError, match="combined"):
        append_sampler_capability(reuse, NoiseInversionOptions())
    with pytest.raises(ValueError, match="combined"):
        append_sampler_capability(
            SamplerOptions(noise_inversion=NoiseInversionOptions()), NoiseReuseOptions()
        )


def test_missing_capture_does_not_fall_back_to_fresh_noise() -> None:
    """Require a saved request while retaining its independently observed coordinate."""
    state = torch.zeros(1, 4, 2, 2)
    provenance = SamplingProvenance(
        "SDXL", state, (CapturedSamplingState(0.5, 0.5, state, state),)
    )
    assert provenance.state_at(0.5) is state
    with pytest.raises(ValueError, match="did not save"):
        provenance.state_at(0.4)


def test_noise_amplitude_survives_large_clean_reference() -> None:
    """Avoid subtracting near-identical large values to measure noise amplitude."""
    from types import SimpleNamespace

    from simple_syrup.runtime.native_noise_scaling import noise_coefficients

    clean = torch.full((1, 4, 2, 2), 1e10)
    scaling = SimpleNamespace(
        noise_scaling=lambda sigma, noise, reference, maximum: reference + sigma * noise
    )
    base, amplitude = noise_coefficients(scaling, clean, 0.5)
    assert torch.equal(base, clean)
    assert torch.equal(amplitude, torch.full_like(clean, 0.5))
