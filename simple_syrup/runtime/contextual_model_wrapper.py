# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fuse tiled predictions with scheduled reduced-global scene authority."""

from __future__ import annotations

from typing import Any

import torch

from ..domain.contextual_diffusion import (
    ContextualDiffusionControls,
    ContextualDiffusionPlan,
)
from ..domain.global_context_schedule import GlobalContextSchedule
from ..domain.spatial_views import SpatialBatchLayout
from .sampling_model_types import ApplyModel, ModelFunctionWrapper
from .spatial_model_arguments import make_spatial_view_model_args
from .spatial_tensor_projection import resize_spatial_tensor
from .tile_prediction_accumulation import TilePredictionAccumulator
from .tiled_sampling_validation import validate_tensor_shape


class ContextualDiffusionModelWrapper:
    """Fuse one authoritative tiled prediction with scheduled global structure."""

    def __init__(
        self,
        *,
        plan: ContextualDiffusionPlan,
        controls: ContextualDiffusionControls,
        sigmas: torch.Tensor,
        diffusion_mode: str = "multidiffusion",
        existing_wrapper: ModelFunctionWrapper | None,
    ) -> None:
        """Create a wrapper for one immutable latent and context plan."""

        self._plan = plan
        self._controls = controls
        self._global_schedule = GlobalContextSchedule(
            sigmas=sigmas,
            active_steps=controls.global_steps,
            decay=controls.global_decay,
        )
        self._existing_wrapper = existing_wrapper
        self._diffusion_mode = diffusion_mode
        self._tile_predictions = TilePredictionAccumulator(
            plan.tile_plan,
            diffusion_mode=diffusion_mode,
        )

    @property
    def diffusion_mode(self) -> str:
        """Return the selected local tile blending policy."""

        return self._diffusion_mode

    def __call__(self, apply_model: ApplyModel, args: dict[str, Any]) -> torch.Tensor:
        """Evaluate bounded contexts and return one canvas-sized prediction."""

        x = args.get("input")
        if not isinstance(x, torch.Tensor):
            raise ValueError("Contextual Diffusion model input must be a tensor.")
        validate_tensor_shape(x, sampler_label="Contextual Diffusion")
        if x.shape[-2:] != (self._plan.latent_height, self._plan.latent_width):
            return self._call_original(apply_model, args)
        if len(self._plan.tile_plan.tiles) == 1:
            return self._call_original(apply_model, args)

        tile_prediction = self._predict_tiles(apply_model, args, x)
        global_scale = self._global_schedule.scale_for(args.get("timestep"))
        if global_scale <= 0 or self._controls.global_weight <= 0:
            return tile_prediction
        return self._apply_global_authority(
            apply_model,
            args,
            tile_prediction,
            weight=self._controls.global_weight * global_scale,
        )

    def _predict_tiles(
        self, apply_model: ApplyModel, args: dict[str, Any], x: torch.Tensor
    ) -> torch.Tensor:
        """Blend the authoritative regular or SEGS-guided tiled prediction."""

        return self._tile_predictions.predict(
            args=args,
            x=x,
            evaluate=lambda tiled_args: self._call_original(apply_model, tiled_args),
        )

    def _apply_global_authority(
        self,
        apply_model: ApplyModel,
        args: dict[str, Any],
        local_prediction: torch.Tensor,
        *,
        weight: float,
    ) -> torch.Tensor:
        """Replace tile-scale scene interpretation with the whole-image prediction."""

        global_layout = SpatialBatchLayout(
            canvas_width=self._plan.latent_width,
            canvas_height=self._plan.latent_height,
            views=(self._plan.global_view,),
            input_batch_size=int(local_prediction.shape[0]),
        )
        global_args = make_spatial_view_model_args(
            args=args,
            layout=global_layout,
        )
        global_prediction = self._call_original(apply_model, global_args)
        global_view = self._plan.global_view
        global_canvas = resize_spatial_tensor(
            global_prediction,
            height=self._plan.latent_height,
            width=self._plan.latent_width,
            mode="bilinear",
        )
        local_low = resize_spatial_tensor(
            local_prediction,
            height=global_view.model_height,
            width=global_view.model_width,
            mode="area",
        )
        local_low = resize_spatial_tensor(
            local_low,
            height=self._plan.latent_height,
            width=self._plan.latent_width,
            mode="bilinear",
        )
        return local_prediction + weight * (global_canvas - local_low)

    def _call_original(
        self, apply_model: ApplyModel, args: dict[str, Any]
    ) -> torch.Tensor:
        """Call the preserved model wrapper or raw apply_model."""

        if self._existing_wrapper is not None:
            return self._existing_wrapper(apply_model, args)
        conditioning = args.get("c", {})
        if not isinstance(conditioning, dict):
            raise ValueError("Contextual Diffusion conditioning must be a dict.")
        return apply_model(args["input"], args["timestep"], **conditioning)
