# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Normalize latent tensors through ComfyUI's model-aware host contract."""

from __future__ import annotations

from importlib import import_module

import torch


class ComfyLatentNormalizer:
    """Apply ComfyUI's channel, spatial, and temporal latent normalization."""

    def normalize(
        self,
        *,
        model: object,
        samples: torch.Tensor,
        spatial_downscale_ratio: object | None = None,
        temporal_downscale_ratio: object | None = None,
    ) -> torch.Tensor:
        """Return samples in the latent layout required by the supplied model."""

        normalize = import_module("comfy.sample").fix_empty_latent_channels
        if temporal_downscale_ratio is None:
            normalized = normalize(model, samples, spatial_downscale_ratio)
        else:
            normalized = normalize(
                model,
                samples,
                spatial_downscale_ratio,
                temporal_downscale_ratio,
            )
        if not isinstance(normalized, torch.Tensor):
            raise TypeError("ComfyUI latent normalization must return a torch.Tensor.")
        return normalized


COMFY_LATENT_NORMALIZER = ComfyLatentNormalizer()
