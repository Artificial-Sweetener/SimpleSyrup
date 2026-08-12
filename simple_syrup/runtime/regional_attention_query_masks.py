# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Project canonical masks into backend-neutral query-batch order."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..domain.regional_mask_bank import RegionalMaskBank
from ..domain.spatial_views import SpatialBatchLayout
from ..masking.regional_mask_projection import (
    RegionalMaskForm,
    RegionalMaskProjectionMode,
    RegionalMaskProjector,
)
from .regional_attention_diagnostic_values import RegionalAttentionQueryGeometry


@dataclass(frozen=True, slots=True)
class RegionalAttentionQueryMaskBatch:
    """Retain canonical R/B/H/W and flattened R/B/Q query masks."""

    masks: torch.Tensor

    def __post_init__(self) -> None:
        """Require non-empty finite floating query masks."""

        if not isinstance(self.masks, torch.Tensor):
            raise TypeError("Regional query masks must be a tensor.")
        if self.masks.ndim != 4 or any(int(size) < 1 for size in self.masks.shape):
            raise ValueError("Regional query masks must use non-empty R/B/H/W layout.")
        if not self.masks.is_floating_point():
            raise TypeError("Regional query masks must use a floating dtype.")
        if not bool(torch.isfinite(self.masks).all()):
            raise ValueError("Regional query masks must contain finite values.")

    @property
    def flattened(self) -> torch.Tensor:
        """Return the same masks in R/B/Q attention-token layout."""

        return self.masks.flatten(start_dim=2)


class RegionalAttentionQueryMaskProjector:
    """Own exact view/chunk/latent expansion around canonical projection."""

    def __init__(self, projector: RegionalMaskProjector | None = None) -> None:
        """Retain the sole canonical crop and interpolation collaborator."""

        self._projector = projector or RegionalMaskProjector()

    def project(
        self,
        *,
        bank: RegionalMaskBank,
        query_geometry: RegionalAttentionQueryGeometry,
        layout: SpatialBatchLayout,
        form: RegionalMaskForm,
        mode: RegionalMaskProjectionMode,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RegionalAttentionQueryMaskBatch:
        """Return masks in the exact active model query-batch order."""

        if not isinstance(bank, RegionalMaskBank):
            raise TypeError("Regional query projection requires a mask bank.")
        if not isinstance(query_geometry, RegionalAttentionQueryGeometry):
            raise TypeError("Regional query projection requires query geometry.")
        if not isinstance(layout, SpatialBatchLayout):
            raise TypeError("Regional query projection requires a spatial layout.")
        if not isinstance(form, RegionalMaskForm):
            raise TypeError("Regional query mask form has an invalid type.")
        if not isinstance(mode, RegionalMaskProjectionMode):
            raise TypeError("Regional query projection mode has an invalid type.")
        if not isinstance(device, torch.device):
            raise TypeError("Regional query mask device must be a torch.device.")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("Regional query mask dtype must be floating point.")
        projected_views = tuple(
            self._projector.project_query_grid(
                bank=bank,
                layout=layout,
                view_index=view_index,
                query_height=query_geometry.query_height,
                query_width=query_geometry.query_width,
                form=form,
                mode=mode,
            )
            for view_index in range(layout.view_count)
        )
        view_major = torch.cat(
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
        )
        if query_geometry.spatial_layout is None:
            if layout.view_count != 1:
                raise ValueError(
                    "Implicit regional query layout must contain one full view."
                )
            if query_geometry.input_batch_size % layout.expanded_batch_size != 0:
                raise ValueError(
                    "Implicit regional query batch must divide over its full layout."
                )
            repeat_count = query_geometry.input_batch_size // layout.expanded_batch_size
            batched = view_major.repeat(1, repeat_count, 1, 1)
        else:
            if layout is not query_geometry.spatial_layout:
                raise ValueError(
                    "Regional query layout must be the published geometry authority."
                )
            batched = view_major
        if int(batched.shape[1]) != query_geometry.input_batch_size:
            raise ValueError(
                "Regional query mask batch must match query geometry batch."
            )
        return RegionalAttentionQueryMaskBatch(batched.to(device=device, dtype=dtype))


REGIONAL_ATTENTION_QUERY_MASK_PROJECTOR = RegionalAttentionQueryMaskProjector()
