# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for structural FLUX generation and encoder-profile classification."""

from __future__ import annotations

import pytest

from simple_syrup.domain.flux_profiles import (
    Flux2TextEncoderProfile,
    FluxGeneration,
    classify_flux_profile,
)


def test_classifies_flux1_without_filename_or_parameter_count() -> None:
    """ComfyUI's FLUX image-model marker is sufficient for FLUX.1."""

    profile = classify_flux_profile("flux", 4_096)

    assert profile is not None
    assert profile.generation is FluxGeneration.FLUX
    assert profile.flux2_text_encoder is None


@pytest.mark.parametrize(
    ("context_dimension", "expected"),
    (
        (15_360, Flux2TextEncoderProfile.DEV),
        (7_680, Flux2TextEncoderProfile.KLEIN_4B),
        (12_288, Flux2TextEncoderProfile.KLEIN_9B),
    ),
)
def test_classifies_flux2_encoder_from_conditioning_dimension(
    context_dimension: int,
    expected: Flux2TextEncoderProfile,
) -> None:
    """FLUX.2 dev and Klein variants map from tensor-derived input width."""

    profile = classify_flux_profile("flux2", context_dimension)

    assert profile is not None
    assert profile.generation is FluxGeneration.FLUX2
    assert profile.flux2_text_encoder is expected


def test_preserves_flux2_generation_for_unknown_conditioning_dimension() -> None:
    """Unknown future FLUX.2 widths remain FLUX.2 with unresolved encoder policy."""

    profile = classify_flux_profile("flux2", 99_999)

    assert profile is not None
    assert profile.generation is FluxGeneration.FLUX2
    assert profile.flux2_text_encoder is None


def test_rejects_non_flux_image_model_marker() -> None:
    """Unrelated ComfyUI model families do not receive a FLUX profile."""

    assert classify_flux_profile("sd3", 4_096) is None
