# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plan one bounded global context and one authoritative tiled context set."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .regional_tiled_diffusion import build_region_constrained_tiled_diffusion_plan
from .segs import NativeSegs
from .segs_tiled_diffusion import build_segs_guided_tiled_diffusion_plan
from .tiled_diffusion import TiledDiffusionPlan, build_tiled_diffusion_plan


@dataclass(frozen=True)
class ContextualDiffusionControls:
    """Validate workflow controls for contextual diffusion sampling."""

    latent_context_size: int
    latent_context_overlap: int
    latent_context_batch_size: int
    global_weight: float
    global_steps: int
    global_decay: float

    def validate(self) -> None:
        """Reject controls that cannot produce a stable bounded context plan."""

        if self.latent_context_size < 16:
            raise ValueError("latent_context_size must be at least 16 latent pixels.")
        if not 0 <= self.latent_context_overlap < self.latent_context_size:
            raise ValueError(
                "latent_context_overlap must be non-negative and smaller than "
                "latent_context_size."
            )
        if self.latent_context_batch_size < 1:
            raise ValueError("latent_context_batch_size must be at least 1.")
        if not 0.0 <= self.global_weight <= 2.0:
            raise ValueError("global_weight must be between 0 and 2.")
        if self.global_steps < 0:
            raise ValueError("global_steps must be non-negative.")
        if not 0.0 <= self.global_decay <= 1.0:
            raise ValueError("global_decay must be between 0 and 1.")


@dataclass(frozen=True)
class SpatialContext:
    """Describe one source rectangle evaluated at a bounded model context shape."""

    x: int
    y: int
    width: int
    height: int
    context_width: int
    context_height: int


@dataclass(frozen=True)
class ContextualDiffusionPlan:
    """Own the global context and sole tiled plan for one latent canvas."""

    latent_width: int
    latent_height: int
    global_context: SpatialContext
    tile_plan: TiledDiffusionPlan


def build_contextual_diffusion_plan(
    *,
    latent_width: int,
    latent_height: int,
    controls: ContextualDiffusionControls,
    segs: NativeSegs | None,
    region_masks: torch.Tensor | None = None,
) -> ContextualDiffusionPlan:
    """Return a global context plus the regular or SEGS-guided context plan."""

    controls.validate()
    global_width, global_height = fit_context_shape(
        latent_width,
        latent_height,
        controls.latent_context_size,
    )
    global_context = SpatialContext(
        x=0,
        y=0,
        width=latent_width,
        height=latent_height,
        context_width=global_width,
        context_height=global_height,
    )
    if region_masks is not None:
        tile_plan = build_region_constrained_tiled_diffusion_plan(
            region_masks=region_masks,
            segs=segs,
            latent_width=latent_width,
            latent_height=latent_height,
            tile_width=controls.latent_context_size,
            tile_height=controls.latent_context_size,
            overlap=controls.latent_context_overlap,
            tile_batch_size=controls.latent_context_batch_size,
        )
    elif segs is not None:
        tile_plan = build_segs_guided_tiled_diffusion_plan(
            segs=segs,
            latent_width=latent_width,
            latent_height=latent_height,
            tile_width=controls.latent_context_size,
            tile_height=controls.latent_context_size,
            overlap=controls.latent_context_overlap,
            tile_batch_size=controls.latent_context_batch_size,
        )
    else:
        tile_plan = build_tiled_diffusion_plan(
            latent_width=latent_width,
            latent_height=latent_height,
            tile_width=controls.latent_context_size,
            tile_height=controls.latent_context_size,
            overlap=controls.latent_context_overlap,
            tile_batch_size=controls.latent_context_batch_size,
        )
    return ContextualDiffusionPlan(
        latent_width=latent_width,
        latent_height=latent_height,
        global_context=global_context,
        tile_plan=tile_plan,
    )


def fit_context_shape(width: int, height: int, max_size: int) -> tuple[int, int]:
    """Fit a rectangle inside one maximum latent dimension without boxing it."""

    if width < 1 or height < 1:
        raise ValueError("Context source dimensions must be positive.")
    if max_size < 1:
        raise ValueError("Context maximum size must be positive.")
    if max(width, height) <= max_size:
        return width, height
    scale = max_size / max(width, height)
    fitted_width = max(2, round(width * scale))
    fitted_height = max(2, round(height * scale))
    return _even_at_most(fitted_width, max_size), _even_at_most(
        fitted_height,
        max_size,
    )


def _even_at_most(value: int, maximum: int) -> int:
    """Return a positive even model-context dimension within its maximum."""

    bounded = min(maximum, max(2, value))
    if bounded % 2 == 0:
        return bounded
    if bounded == maximum:
        return max(2, bounded - 1)
    return bounded + 1
