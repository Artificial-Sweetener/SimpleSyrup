# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for trusted FLUX automatic artifact metadata."""

from __future__ import annotations

from simple_syrup.domain.flux_profiles import Flux2TextEncoderProfile
from simple_syrup.runtime.flux_artifacts import (
    FLUX2_DEV_TEXT_ENCODER,
    FLUX2_KLEIN_4B_TEXT_ENCODER,
    FLUX2_KLEIN_9B_TEXT_ENCODER,
    FLUX2_TEXT_ENCODERS,
    FLUX2_VAE,
    FLUX_CLIP_L,
    FLUX_T5_XXL,
    FLUX_VAE,
)


def test_flux_artifacts_are_checksum_pinned_to_immutable_revisions() -> None:
    """Every automatic file has a SHA-256 and commit-pinned Hugging Face URL."""

    artifacts = (
        FLUX_CLIP_L,
        FLUX_T5_XXL,
        FLUX_VAE,
        FLUX2_DEV_TEXT_ENCODER,
        FLUX2_KLEIN_4B_TEXT_ENCODER,
        FLUX2_KLEIN_9B_TEXT_ENCODER,
        FLUX2_VAE,
    )

    for artifact in artifacts:
        assert artifact.source_url.startswith("https://huggingface.co/")
        assert "/resolve/main/" not in artifact.source_url
        assert len(artifact.sha256) == 64
        assert set(artifact.sha256) <= set("0123456789abcdef")


def test_flux2_encoder_profiles_map_to_the_expected_families() -> None:
    """Dev and both Klein architecture widths select distinct encoders."""

    assert FLUX2_TEXT_ENCODERS == {
        Flux2TextEncoderProfile.DEV: FLUX2_DEV_TEXT_ENCODER,
        Flux2TextEncoderProfile.KLEIN_4B: FLUX2_KLEIN_4B_TEXT_ENCODER,
        Flux2TextEncoderProfile.KLEIN_9B: FLUX2_KLEIN_9B_TEXT_ENCODER,
    }
    assert FLUX2_DEV_TEXT_ENCODER.filename.startswith("mistral_3_small")
    assert FLUX2_KLEIN_4B_TEXT_ENCODER.filename == "qwen_3_4b.safetensors"
    assert FLUX2_KLEIN_9B_TEXT_ENCODER.filename == ("qwen_3_8b_fp8mixed.safetensors")


def test_flux_generations_use_separate_vae_artifacts() -> None:
    """FLUX.1 and FLUX.2 auto select their generation-specific VAE files."""

    assert FLUX_VAE.filename == "ae.safetensors"
    assert FLUX2_VAE.filename == "flux2-vae.safetensors"
