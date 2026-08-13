# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Project canonical masks into exact regional activation multiplier shapes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..domain.regional_activation_geometry import (
    RegionalActivationGeometry,
    RegionalActivationLayout,
)
from ..domain.regional_mask_bank import RegionalMaskBank
from ..domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from .regional_mask_projection import (
    RegionalMaskForm,
    RegionalMaskProjectionMode,
    RegionalMaskProjector,
)


@dataclass(frozen=True, slots=True)
class RegionalActivationMaskBatch:
    """Retain one finite region-major multiplier broadcast over rank activations."""

    multipliers: torch.Tensor
    geometry: RegionalActivationGeometry

    def __post_init__(self) -> None:
        """Require exact geometry shape and bounded finite floating multipliers."""

        if not isinstance(self.multipliers, torch.Tensor):
            raise TypeError("Regional activation multipliers must be a tensor.")
        if not isinstance(self.geometry, RegionalActivationGeometry):
            raise TypeError("Regional activation multipliers require geometry.")
        if not self.multipliers.is_floating_point():
            raise TypeError("Regional activation multipliers must use floating point.")
        if self.multipliers.ndim != len(self.geometry.invocation_shape) + 1:
            raise ValueError("Regional activation multiplier rank is inconsistent.")
        expected = self.geometry.broadcast_mask_shape(int(self.multipliers.shape[0]))
        if tuple(self.multipliers.shape) != expected:
            raise ValueError(
                "Regional activation multiplier shape must match its geometry."
            )
        if not bool(torch.isfinite(self.multipliers).all()):
            raise ValueError("Regional activation multipliers must be finite.")
        if not bool(((self.multipliers >= 0.0) & (self.multipliers <= 1.0)).all()):
            raise ValueError("Regional activation multipliers must stay within [0, 1].")


class RegionalActivationMaskProjector:
    """Own geometry-shaped mask projection around existing crop/interpolation."""

    def __init__(self, projector: RegionalMaskProjector | None = None) -> None:
        """Retain the canonical mask crop and interpolation authority."""

        self._projector = projector or RegionalMaskProjector()

    def project(
        self,
        *,
        bank: RegionalMaskBank,
        geometry: RegionalActivationGeometry,
        form: RegionalMaskForm,
        mode: RegionalMaskProjectionMode,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RegionalActivationMaskBatch:
        """Return masks in region/view/chunk/latent activation order."""

        _validate_inputs(bank, geometry, form, mode, device, dtype)
        layout = geometry.batch_alignment.spatial_layout or _full_layout(
            bank,
            input_batch_size=geometry.batch_alignment.base_batch_size,
        )
        projected_views = tuple(
            self._projector.project_query_grid(
                bank=bank,
                layout=layout,
                view_index=view_index,
                query_height=geometry.spatial_height,
                query_width=geometry.spatial_width,
                form=form,
                mode=mode,
            )
            for view_index in range(layout.view_count)
        )
        masks = torch.cat(
            tuple(
                projected.unsqueeze(1).expand(
                    -1,
                    layout.input_batch_size,
                    -1,
                    -1,
                )
                for projected in projected_views
            ),
            dim=1,
        ).to(device=device, dtype=dtype)
        multipliers = _reshape_for_activation(masks, geometry)
        return RegionalActivationMaskBatch(multipliers, geometry)


def _reshape_for_activation(
    masks: torch.Tensor,
    geometry: RegionalActivationGeometry,
) -> torch.Tensor:
    """Place projected H/W masks on the declared operation's non-feature axes."""

    regions, batch, height, width = (int(value) for value in masks.shape)
    if geometry.layout is RegionalActivationLayout.DIRECT_CONVOLUTION_1D:
        if height != 1:
            raise ValueError("Conv1d regional masks require projected height one.")
        return masks.reshape(regions, batch, 1, width)
    if geometry.layout is RegionalActivationLayout.DIRECT_CONVOLUTION_2D:
        return masks.reshape(regions, batch, 1, height, width)
    if geometry.layout is RegionalActivationLayout.DIRECT_CONVOLUTION_3D:
        temporal_size = geometry.temporal_size
        if temporal_size is None:
            raise ValueError("Conv3d regional masks require explicit temporal size.")
        return masks.reshape(regions, batch, 1, 1, height, width).expand(
            -1,
            -1,
            -1,
            temporal_size,
            -1,
            -1,
        )
    if geometry.layout in (
        RegionalActivationLayout.FLATTENED_SPATIAL_TOKENS,
        RegionalActivationLayout.CONSUMER_SPATIALIZED,
    ):
        return masks.flatten(start_dim=2).unsqueeze(-1)
    raise AssertionError(f"Unhandled regional activation layout: {geometry.layout}")


def _full_layout(
    bank: RegionalMaskBank,
    *,
    input_batch_size: int,
) -> SpatialBatchLayout:
    """Represent one full-canvas invocation through the shared layout contract."""

    return SpatialBatchLayout(
        bank.canvas_width,
        bank.canvas_height,
        (
            SpatialView(
                SpatialViewKind.FULL,
                0,
                0,
                bank.canvas_width,
                bank.canvas_height,
                bank.canvas_width,
                bank.canvas_height,
            ),
        ),
        input_batch_size,
    )


def _validate_inputs(
    bank: object,
    geometry: object,
    form: object,
    mode: object,
    device: object,
    dtype: object,
) -> None:
    """Validate typed projection inputs without moving or mutating source masks."""

    if not isinstance(bank, RegionalMaskBank):
        raise TypeError("Regional activation projection requires a mask bank.")
    if not isinstance(geometry, RegionalActivationGeometry):
        raise TypeError("Regional activation projection requires geometry.")
    if not isinstance(form, RegionalMaskForm):
        raise TypeError("Regional activation mask form has an invalid type.")
    if not isinstance(mode, RegionalMaskProjectionMode):
        raise TypeError("Regional activation projection mode has an invalid type.")
    if not isinstance(device, torch.device):
        raise TypeError("Regional activation mask device must be a torch.device.")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("Regional activation mask dtype must be floating point.")


REGIONAL_ACTIVATION_MASK_PROJECTOR = RegionalActivationMaskProjector()
