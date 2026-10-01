# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026 Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare each inversion resolution from the original, spatially unwrapped MODEL."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as functional

from ..domain.contextual_diffusion import (
    ContextualDiffusionControls,
    build_contextual_diffusion_plan,
)
from ..domain.regional_tiled_diffusion import (
    build_region_constrained_tiled_diffusion_plan,
)
from ..domain.sampler_options import TilingOptions
from ..domain.segs import NativeSegs
from ..domain.segs_tiled_diffusion import build_segs_guided_tiled_diffusion_plan
from ..domain.tiled_diffusion import TiledDiffusionPlan, build_tiled_diffusion_plan
from .inversion_spatial_context import InversionSpatialContextWrapper
from .model_patcher_mutations import ModelUnetWrapperMutation
from .patcher_lifecycle import PATCHER_LIFECYCLE
from .sampling_model_types import ModelFunctionWrapper


class InversionModelFactory:
    """Own stage planning while sharing the forward spatial wrapper authorities."""

    def __init__(
        self,
        *,
        model: Any,
        canvas_width: int,
        canvas_height: int,
        tiling: TilingOptions | None = None,
        context: ContextualDiffusionControls | None = None,
        forward_sigmas: torch.Tensor | None = None,
        segs: NativeSegs | None = None,
        region_masks: torch.Tensor | None = None,
    ) -> None:
        """Retain canonical inputs, not an already wrapped full-resolution model."""
        if context is not None and (tiling is None or forward_sigmas is None):
            raise ValueError(
                "Contextual inversion requires tile controls and forward sigmas."
            )
        self._model = model
        self._width = canvas_width
        self._height = canvas_height
        self._tiling = tiling
        self._context = context
        self._sigmas = forward_sigmas
        self._segs = segs
        self._masks = region_masks

    def __call__(self, latent: torch.Tensor) -> Any:
        """Replan one resolution with canonical regional-mask coordinates."""
        from .contextual_diffusion_sampling import clone_model_with_contextual_diffusion
        from .mixture_of_diffusers_sampling import clone_model_with_mixture_of_diffusers
        from .multidiffusion_sampling import clone_model_with_multidiffusion

        width, height = int(latent.shape[-1]), int(latent.shape[-2])
        old_wrapper = self._model.model_options.get("model_function_wrapper")
        if old_wrapper is not None and not callable(old_wrapper):
            raise TypeError("Existing model_function_wrapper must be callable.")
        wrapper = cast(ModelFunctionWrapper | None, old_wrapper)
        if wrapper is not None and (width, height) != (self._width, self._height):
            wrapper = InversionSpatialContextWrapper(
                wrapper,
                canvas_width=self._width,
                canvas_height=self._height,
                stage_width=width,
                stage_height=height,
            )
        masks = self._stage_masks(height, width)
        if self._context is not None:
            assert self._tiling is not None and self._sigmas is not None
            plan = build_contextual_diffusion_plan(
                latent_width=width,
                latent_height=height,
                controls=self._context,
                segs=self._segs,
                region_masks=masks,
                segs_canvas=(self._height, self._width),
            )
            return clone_model_with_contextual_diffusion(
                self._model,
                plan=plan,
                controls=self._context,
                sigmas=self._sigmas,
                diffusion_mode=self._tiling.diffusion_mode,
                differential_diffusion=self._tiling.differential_diffusion,
                existing_wrapper=wrapper,
            )
        if self._tiling is not None:
            tile_plan = self._tile_plan(width, height, masks)
            clone = (
                clone_model_with_multidiffusion
                if self._tiling.diffusion_mode == "multidiffusion"
                else clone_model_with_mixture_of_diffusers
            )
            derived, _ = clone(
                self._model,
                latent_width=width,
                latent_height=height,
                tile_width=self._tiling.width,
                tile_height=self._tiling.height,
                overlap=self._tiling.overlap,
                tile_batch_size=self._tiling.batch_size,
                differential_diffusion=self._tiling.differential_diffusion,
                tiled_plan=tile_plan,
                existing_wrapper=wrapper,
            )
            return derived
        if wrapper is old_wrapper:
            return self._model
        assert wrapper is not None
        return PATCHER_LIFECYCLE.derive_model(
            self._model,
            (ModelUnetWrapperMutation(wrapper),),
            operation="SimpleSyrup inversion spatial context",
        )

    def _stage_masks(self, height: int, width: int) -> torch.Tensor | None:
        """Resize planning masks once; attention masks keep their canonical bank."""
        if self._masks is None:
            return None
        if tuple(self._masks.shape[-2:]) == (height, width):
            return self._masks
        return functional.interpolate(
            self._masks.unsqueeze(1).float(), size=(height, width), mode="nearest"
        ).squeeze(1)

    def _tile_plan(
        self, width: int, height: int, masks: torch.Tensor | None
    ) -> TiledDiffusionPlan:
        """Share regular, SEGS and regional ownership with forward sampling."""
        assert self._tiling is not None
        geometry = {
            "latent_width": width,
            "latent_height": height,
            "tile_width": self._tiling.width,
            "tile_height": self._tiling.height,
            "overlap": self._tiling.overlap,
            "tile_batch_size": self._tiling.batch_size,
        }
        if masks is not None:
            return build_region_constrained_tiled_diffusion_plan(
                region_masks=masks,
                segs=self._segs,
                segs_canvas=(self._height, self._width),
                **geometry,
            )
        if self._segs is not None:
            return build_segs_guided_tiled_diffusion_plan(
                segs=self._segs,
                segs_canvas=(self._height, self._width),
                **geometry,
            )
        return build_tiled_diffusion_plan(**geometry)
