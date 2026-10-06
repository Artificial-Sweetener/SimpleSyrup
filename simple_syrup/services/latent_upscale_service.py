# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Own latent upscale method selection, provenance preservation and mask geometry."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from ..domain.latent_upscale import (
    FORMAT_FAMILIES,
    NATIVE_LATENT_METHODS,
    upscale_target,
)
from ..domain.noise_reuse import PROVENANCE_KEY, SamplingProvenance
from ..runtime.latent_upscale_models import resolve_latent_upscale_model
from ..runtime.sesqui_latent_upscale import upscale_with_sesqui
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)


class LatentUpscaleService:
    """Resolve learned checkpoints or delegate native scaling to Comfy's node."""

    def upscale(
        self, latent: dict[str, Any], method: str, scale_factor: float
    ) -> dict[str, Any]:
        """Use native bislerp for unknown Auto formats and preserve provenance."""
        samples = latent.get("samples")
        if not isinstance(samples, torch.Tensor) or samples.ndim not in (4, 5):
            raise ValueError("Simple Upscale Latent requires BCHW or BCTHW samples.")
        if not samples.is_floating_point() or not bool(torch.isfinite(samples).all()):
            raise ValueError(
                "Simple Upscale Latent requires finite floating-point samples."
            )
        mask = latent.get("noise_mask")
        if mask is not None and (not isinstance(mask, torch.Tensor) or mask.ndim < 2):
            raise TypeError("Latent noise mask must contain spatial tensor dimensions.")
        target = upscale_target(
            (int(samples.shape[-2]), int(samples.shape[-1])), scale_factor
        )
        if method == "Auto":
            provenance = latent.get(PROVENANCE_KEY)
            latent_format = (
                provenance.latent_format
                if isinstance(provenance, SamplingProvenance)
                else None
            )
            if latent_format not in FORMAT_FAMILIES:
                method = "bislerp"
                LOGGER.info(
                    "Simple Upscale Latent selected native bislerp: "
                    "latent_format=%s scale_factor=%s",
                    latent_format or "unknown",
                    scale_factor,
                )
        if method in NATIVE_LATENT_METHODS:
            result: dict[str, Any] = (
                import_module("nodes")
                .LatentUpscaleBy()
                .upscale(latent, method, scale_factor)[0]
            )
        else:
            family, path = resolve_latent_upscale_model(latent, method)
            result = latent.copy()
            result["samples"] = upscale_with_sesqui(
                samples, path=path, family=family, target=target
            )
        if mask is not None:
            assert isinstance(mask, torch.Tensor)
            resized = torch.nn.functional.interpolate(
                mask.reshape(-1, 1, *mask.shape[-2:]).float(),
                size=target,
                mode="nearest",
            )
            result["noise_mask"] = resized.reshape(*mask.shape[:-2], *target).to(mask)
        return result
