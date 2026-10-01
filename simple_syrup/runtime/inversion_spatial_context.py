# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Project reduced inversion views into the original regional-attention canvas."""

from __future__ import annotations

from typing import Any

import torch

from ..domain.spatial_views import SpatialBatchLayout, SpatialView, SpatialViewKind
from .sampling_model_types import ApplyModel, ModelFunctionWrapper
from .spatial_model_arguments import (
    SIMPLE_SYRUP_TRANSFORMER_NAMESPACE,
    SPATIAL_BATCH_LAYOUT_KEY,
)


def rebase_inversion_layout(
    layout: SpatialBatchLayout, *, canvas_width: int, canvas_height: int
) -> SpatialBatchLayout:
    """Keep actual model dimensions while mapping source rectangles to full masks."""
    scale_x = canvas_width / layout.canvas_width
    scale_y = canvas_height / layout.canvas_height
    views: list[SpatialView] = []
    for view in layout.views:
        left = round(view.source_x * scale_x)
        top = round(view.source_y * scale_y)
        right = round(view.source_right * scale_x)
        bottom = round(view.source_bottom * scale_y)
        views.append(
            SpatialView(
                kind=(
                    SpatialViewKind.CONTEXTUAL_GLOBAL
                    if view.kind is SpatialViewKind.FULL
                    else view.kind
                ),
                source_x=left,
                source_y=top,
                source_width=right - left,
                source_height=bottom - top,
                model_width=view.model_width,
                model_height=view.model_height,
            )
        )
    return SpatialBatchLayout(
        canvas_width, canvas_height, tuple(views), layout.input_batch_size
    )


class InversionSpatialContextWrapper:
    """Preserve regional mask coordinates without resizing conditioning twice."""

    def __init__(
        self,
        existing_wrapper: ModelFunctionWrapper,
        *,
        canvas_width: int,
        canvas_height: int,
        stage_width: int,
        stage_height: int,
    ) -> None:
        """Bind one stage's geometry to the original attention-mask canvas."""
        self._existing_wrapper = existing_wrapper
        self._canvas_width = canvas_width
        self._canvas_height = canvas_height
        self._stage_width = stage_width
        self._stage_height = stage_height

    def __call__(self, apply_model: ApplyModel, args: dict[str, Any]) -> torch.Tensor:
        """Replace only layout metadata, preserving expanded CFG batch metadata."""
        x = args.get("input")
        c = args.get("c", {})
        if not isinstance(x, torch.Tensor) or not isinstance(c, dict):
            raise TypeError(
                "Inversion spatial context requires tensor input and conditioning."
            )
        transformer_options = c.get("transformer_options", {})
        if not isinstance(transformer_options, dict):
            raise TypeError("Inversion transformer_options must be a dictionary.")
        namespace = transformer_options.get(SIMPLE_SYRUP_TRANSFORMER_NAMESPACE, {})
        if not isinstance(namespace, dict):
            raise TypeError(
                "Inversion SimpleSyrup transformer namespace must be a dictionary."
            )
        layout = namespace.get(SPATIAL_BATCH_LAYOUT_KEY)
        if layout is None:
            if tuple(x.shape[-2:]) != (self._stage_height, self._stage_width):
                raise ValueError(
                    "Reduced inversion calls require explicit spatial layout."
                )
            layout = SpatialBatchLayout(
                self._stage_width,
                self._stage_height,
                (
                    SpatialView(
                        SpatialViewKind.FULL,
                        0,
                        0,
                        self._stage_width,
                        self._stage_height,
                        self._stage_width,
                        self._stage_height,
                    ),
                ),
                int(x.shape[0]),
            )
        if not isinstance(layout, SpatialBatchLayout):
            raise TypeError("Inversion spatial layout has an invalid type.")
        if (layout.canvas_width, layout.canvas_height) != (
            self._stage_width,
            self._stage_height,
        ) or layout.expanded_batch_size != int(x.shape[0]):
            raise ValueError(
                "Inversion layout must describe the current stage and model batch."
            )
        rebased = rebase_inversion_layout(
            layout, canvas_width=self._canvas_width, canvas_height=self._canvas_height
        )
        projected_namespace = {**namespace, SPATIAL_BATCH_LAYOUT_KEY: rebased}
        projected_options = {
            **transformer_options,
            SIMPLE_SYRUP_TRANSFORMER_NAMESPACE: projected_namespace,
        }
        projected_args = {**args, "c": {**c, "transformer_options": projected_options}}
        return self._existing_wrapper(apply_model, projected_args)
