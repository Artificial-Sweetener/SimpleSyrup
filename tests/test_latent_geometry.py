# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for decoded image geometry derived from Comfy latent metadata."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from simple_syrup.runtime.latent_geometry import decoded_image_dimensions


@dataclass(frozen=True)
class _LatentFormat:
    """Expose ComfyUI's intentionally misspelled spatial ratio attribute."""

    spacial_downscale_ratio: int


class _Model:
    """Provide one latent format through the ModelPatcher boundary."""

    def get_model_object(self, name: str) -> object:
        """Return the requested latent format fixture."""

        assert name == "latent_format"
        return _LatentFormat(8)


def test_decoded_dimensions_prefer_explicit_latent_ratio() -> None:
    """Per-latent metadata overrides the model format for generated geometry."""

    assert decoded_image_dimensions(
        model=_Model(),
        latent_image={"downscale_ratio_spacial": 16},
        latent_height=32,
        latent_width=48,
    ) == (512, 768)


def test_decoded_dimensions_fall_back_to_model_latent_format() -> None:
    """Ordinary latents use the model's authoritative spatial scale."""

    assert decoded_image_dimensions(
        model=_Model(),
        latent_image={},
        latent_height=32,
        latent_width=48,
    ) == (256, 384)


def test_decoded_dimensions_reject_missing_geometry_source() -> None:
    """A model-neutral context output fails clearly when scale is unknowable."""

    with pytest.raises(ValueError, match="latent format"):
        decoded_image_dimensions(
            model=object(),
            latent_image={},
            latent_height=32,
            latent_width=48,
        )
