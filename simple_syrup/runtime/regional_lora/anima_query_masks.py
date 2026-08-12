# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Project canonical masks into active Anima query-batch order."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...domain.regional_mask_bank import RegionalMaskBank
from ...domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from ...masking.regional_mask_projection import (
    RegionalMaskForm,
    RegionalMaskProjectionMode,
)
from ..regional_attention_diagnostic_values import RegionalAttentionQueryGeometry
from ..regional_attention_query_masks import (
    REGIONAL_ATTENTION_QUERY_MASK_PROJECTOR,
    RegionalAttentionQueryMaskProjector,
)
from .anima_activation_context import AnimaActivationGeometry


@dataclass(frozen=True, slots=True)
class AnimaQueryMaskBatch:
    """Retain canonical R/B/T/H/W masks for one active Anima call."""

    masks: torch.Tensor

    def __post_init__(self) -> None:
        """Require non-empty finite floating query masks."""

        if not isinstance(self.masks, torch.Tensor):
            raise TypeError("Anima query masks must be a tensor.")
        if self.masks.ndim != 5 or any(int(size) < 1 for size in self.masks.shape):
            raise ValueError("Anima query masks must use non-empty R/B/T/H/W layout.")
        if not self.masks.is_floating_point():
            raise TypeError("Anima query masks must use a floating dtype.")
        if not bool(torch.isfinite(self.masks).all()):
            raise ValueError("Anima query masks must contain finite values.")

    @property
    def flattened(self) -> torch.Tensor:
        """Return the same masks in R/B/Q attention-token layout."""

        return self.masks.flatten(start_dim=2)


class AnimaQueryMaskProjector:
    """Adapt shared query masks to Anima's singleton temporal axis."""

    def __init__(
        self,
        projector: RegionalAttentionQueryMaskProjector = (
            REGIONAL_ATTENTION_QUERY_MASK_PROJECTOR
        ),
    ) -> None:
        """Retain the model-neutral query-mask batching authority."""

        if not isinstance(projector, RegionalAttentionQueryMaskProjector):
            raise TypeError("Anima query masks require a regional projector.")
        self._projector = projector

    def project(
        self,
        *,
        bank: RegionalMaskBank,
        geometry: AnimaActivationGeometry,
        latent_batch_size: int,
        form: RegionalMaskForm,
        mode: RegionalMaskProjectionMode,
        device: torch.device,
        dtype: torch.dtype,
    ) -> AnimaQueryMaskBatch:
        """Return masks in the active model's view/chunk/latent batch order."""

        if not isinstance(bank, RegionalMaskBank):
            raise TypeError("Anima query projection requires a regional mask bank.")
        if not isinstance(geometry, AnimaActivationGeometry):
            raise TypeError("Anima query projection requires activation geometry.")
        if type(latent_batch_size) is not int or latent_batch_size < 1:
            raise ValueError("Anima query latent batch size must be positive.")
        if not isinstance(form, RegionalMaskForm):
            raise TypeError("Anima query mask form has an invalid type.")
        if not isinstance(mode, RegionalMaskProjectionMode):
            raise TypeError("Anima query mask projection mode has an invalid type.")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("Anima query mask dtype must be floating point.")
        layout = self.resolve_layout(
            bank,
            geometry,
            latent_batch_size=latent_batch_size,
        )
        if geometry.query_time != 1:
            raise ValueError(
                "Anima image query projection requires one temporal query slice."
            )
        shared = self._projector.project(
            bank=bank,
            query_geometry=RegionalAttentionQueryGeometry(
                geometry.input_batch_size,
                geometry.query_time,
                geometry.query_height,
                geometry.query_width,
                geometry.spatial_layout,
            ),
            layout=layout,
            form=form,
            mode=mode,
            device=device,
            dtype=dtype,
        )
        return AnimaQueryMaskBatch(shared.masks.unsqueeze(2))

    def resolve_layout(
        self,
        bank: RegionalMaskBank,
        geometry: AnimaActivationGeometry,
        *,
        latent_batch_size: int,
    ) -> SpatialBatchLayout:
        """Return the exact explicit or synthesized layout used for projection."""

        if not isinstance(bank, RegionalMaskBank):
            raise TypeError("Anima query layout requires a regional mask bank.")
        if not isinstance(geometry, AnimaActivationGeometry):
            raise TypeError("Anima query layout requires activation geometry.")
        if type(latent_batch_size) is not int or latent_batch_size < 1:
            raise ValueError("Anima query layout latent batch size must be positive.")
        return geometry.spatial_layout or self._full_layout(
            bank,
            geometry,
            latent_batch_size=latent_batch_size,
        )

    @staticmethod
    def _full_layout(
        bank: RegionalMaskBank,
        geometry: AnimaActivationGeometry,
        *,
        latent_batch_size: int,
    ) -> SpatialBatchLayout:
        """Represent an unmodified full-canvas call as one authoritative view."""

        if (geometry.activation_height, geometry.activation_width) != (
            bank.canvas_height,
            bank.canvas_width,
        ):
            raise ValueError(
                "Full-context Anima activation H/W must match the regional mask canvas."
            )
        return SpatialBatchLayout(
            canvas_width=bank.canvas_width,
            canvas_height=bank.canvas_height,
            views=(
                SpatialView(
                    SpatialViewKind.FULL,
                    source_x=0,
                    source_y=0,
                    source_width=bank.canvas_width,
                    source_height=bank.canvas_height,
                    model_width=bank.canvas_width,
                    model_height=bank.canvas_height,
                ),
            ),
            input_batch_size=latent_batch_size,
        )


ANIMA_QUERY_MASK_PROJECTOR = AnimaQueryMaskProjector()
