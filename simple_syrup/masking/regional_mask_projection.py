# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Project canonical regional masks into spatial views and query grids."""

from __future__ import annotations

from enum import StrEnum

import torch
import torch.nn.functional as functional

from ..domain.regional_mask_bank import RegionalMaskBank
from ..domain.spatial_views import SpatialBatchLayout, SpatialView


class RegionalMaskForm(StrEnum):
    """Select one canonical regional mask representation."""

    PLANNING = "planning"
    CONDITIONING = "conditioning"


class RegionalMaskProjectionMode(StrEnum):
    """Select the interpolation semantics for one mask projection."""

    CONTINUOUS_COVERAGE = "continuous_coverage"
    SOFT = "soft"
    HARD_PRESERVING = "hard_preserving"
    NEAREST = "nearest"


class RegionalMaskProjector:
    """Own canonical-canvas crop and interpolation policy for regional masks."""

    def project_view(
        self,
        *,
        bank: RegionalMaskBank,
        layout: SpatialBatchLayout,
        view_index: int,
        form: RegionalMaskForm,
        mode: RegionalMaskProjectionMode,
    ) -> torch.Tensor:
        """Project one canonical mask form to a view's model dimensions."""

        view = self._view(bank=bank, layout=layout, view_index=view_index)
        return self.project_query_grid(
            bank=bank,
            layout=layout,
            view_index=view_index,
            query_height=view.model_height,
            query_width=view.model_width,
            form=form,
            mode=mode,
        )

    def project_query_grid(
        self,
        *,
        bank: RegionalMaskBank,
        layout: SpatialBatchLayout,
        view_index: int,
        query_height: int,
        query_width: int,
        form: RegionalMaskForm,
        mode: RegionalMaskProjectionMode,
    ) -> torch.Tensor:
        """Crop one spatial view and resample it directly to an attention grid."""

        if query_height < 1 or query_width < 1:
            raise ValueError("Regional mask query-grid dimensions must be positive.")
        if not isinstance(form, RegionalMaskForm):
            raise TypeError("Regional mask form must be a RegionalMaskForm value.")
        if not isinstance(mode, RegionalMaskProjectionMode):
            raise TypeError(
                "Regional mask projection mode must be a "
                "RegionalMaskProjectionMode value."
            )
        view = self._view(bank=bank, layout=layout, view_index=view_index)
        masks = (
            bank.planning_masks
            if form is RegionalMaskForm.PLANNING
            else bank.conditioning_masks
        )
        cropped = masks[
            :,
            view.source_y : view.source_bottom,
            view.source_x : view.source_right,
        ]
        projected = self._resize(
            cropped,
            height=query_height,
            width=query_width,
            mode=mode,
        ).clamp(0.0, 1.0)
        if not bool(torch.isfinite(projected).all()):
            raise ValueError("Projected regional masks contain non-finite values.")
        return projected

    @staticmethod
    def _view(
        *,
        bank: RegionalMaskBank,
        layout: SpatialBatchLayout,
        view_index: int,
    ) -> SpatialView:
        """Validate canonical canvas identity and return one indexed view."""

        if (
            bank.canvas_width != layout.canvas_width
            or bank.canvas_height != layout.canvas_height
        ):
            raise ValueError(
                "Regional mask bank canvas must match the spatial batch layout."
            )
        if view_index < 0 or view_index >= layout.view_count:
            raise IndexError("Regional mask spatial view index is outside the layout.")
        return layout.views[view_index]

    @staticmethod
    def _resize(
        masks: torch.Tensor,
        *,
        height: int,
        width: int,
        mode: RegionalMaskProjectionMode,
    ) -> torch.Tensor:
        """Apply explicit coverage, soft, or hard-preserving interpolation."""

        source_height = int(masks.shape[-2])
        source_width = int(masks.shape[-1])
        if (source_height, source_width) == (height, width):
            return masks
        batched = masks.unsqueeze(1)
        if mode is RegionalMaskProjectionMode.HARD_PRESERVING:
            return functional.interpolate(
                batched,
                size=(height, width),
                mode="nearest-exact",
            ).squeeze(1)
        if mode is RegionalMaskProjectionMode.NEAREST:
            return functional.interpolate(
                batched,
                size=(height, width),
                mode="nearest",
            ).squeeze(1)
        if mode is RegionalMaskProjectionMode.SOFT:
            return functional.interpolate(
                batched,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        downsample_height = min(source_height, height)
        downsample_width = min(source_width, width)
        projected = batched
        if (downsample_height, downsample_width) != (
            source_height,
            source_width,
        ):
            projected = functional.interpolate(
                projected,
                size=(downsample_height, downsample_width),
                mode="area",
            )
        if tuple(projected.shape[-2:]) != (height, width):
            projected = functional.interpolate(
                projected,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        return projected.squeeze(1)
