# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rebuild regional prediction ownership from canonical inputs for inversion stages."""

from __future__ import annotations

from typing import Any

import torch

from ..domain.regional_detailing import LatentRegion
from ..domain.regional_inversion_geometry import project_inversion_regions
from .inversion_model_factory import InversionModelFactory


class RegionalInversionModelFactory:
    """Retain one original MODEL and the full-resolution region bank."""

    def __init__(
        self,
        *,
        model: Any,
        canvas_width: int,
        canvas_height: int,
        regions: tuple[LatentRegion, ...],
        global_prompt_weight: float,
        differential_diffusion: bool,
    ) -> None:
        """Keep regional conditioning unchanged across source-sized inversion views."""
        self._base = InversionModelFactory(
            model=model,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        self._width = canvas_width
        self._height = canvas_height
        self._regions = regions
        self._weight = global_prompt_weight
        self._differential = differential_diffusion

    def __call__(self, latent: torch.Tensor) -> Any:
        """Use the authoritative regional calc-cond-batch wrapper at each stage size."""
        from .regional_multidiffusion_sampling import (
            clone_model_with_regional_multidiffusion,
        )

        height, width = int(latent.shape[-2]), int(latent.shape[-1])
        regions = project_inversion_regions(
            self._regions,
            source_width=self._width,
            source_height=self._height,
            target_width=width,
            target_height=height,
        )
        derived, _ = clone_model_with_regional_multidiffusion(
            self._base(latent),
            latent_width=width,
            latent_height=height,
            latent_ndim=latent.ndim,
            regions=regions,
            global_prompt_weight=self._weight,
            differential_diffusion=self._differential,
        )
        return derived
