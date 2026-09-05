# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify model-aware latent normalization through the installed ComfyUI host."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from simple_syrup.runtime.comfy_latent_normalization import ComfyLatentNormalizer


class _AnimaModel:
    """Expose the latent-format values used by an Anima model patcher."""

    def get_model_object(self, name: str) -> object:
        """Return a three-dimensional sixteen-channel latent format."""

        if name != "latent_format":
            raise KeyError(name)
        return SimpleNamespace(
            latent_channels=16,
            latent_dimensions=3,
            spacial_downscale_ratio=8,
            temporal_downscale_ratio=4,
            fix_empty_latent=lambda latent: latent,
        )


def test_normalizes_ordinary_empty_image_latent_for_anima() -> None:
    """A standard BCHW empty latent becomes Anima's BC1HW latent."""

    samples = torch.zeros((1, 4, 8, 12))

    normalized = ComfyLatentNormalizer().normalize(
        model=_AnimaModel(),
        samples=samples,
        spatial_downscale_ratio=8,
    )

    assert normalized.shape == (1, 16, 1, 8, 12)
