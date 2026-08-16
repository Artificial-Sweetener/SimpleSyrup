# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Project canonical regional masks into standard-UNet output batches."""

from __future__ import annotations

import torch

from ...domain.regional_mask_bank import RegionalMaskBank
from ...domain.spatial_views import SpatialBatchLayout
from ...masking.regional_mask_projection import (
    RegionalMaskForm,
    RegionalMaskProjectionMode,
    RegionalMaskProjector,
)
from ..spatial_model_arguments import (
    SIMPLE_SYRUP_TRANSFORMER_NAMESPACE,
    SPATIAL_BATCH_LAYOUT_KEY,
)


class StandardUnetVariantMaskProjector:
    """Own full, tiled, and Contextual model-output mask alignment."""

    def __init__(self, projector: RegionalMaskProjector | None = None) -> None:
        """Retain the canonical crop and interpolation authority."""

        self._projector = projector or RegionalMaskProjector()

    def project(
        self,
        *,
        bank: RegionalMaskBank,
        region_indices: tuple[int, ...],
        model_input: torch.Tensor,
        transformer_options: dict[str, object],
    ) -> torch.Tensor:
        """Return selected masks in R/B/1/H/W model-output order."""

        if not isinstance(bank, RegionalMaskBank):
            raise TypeError("Standard UNet variant masks require a mask bank.")
        if not isinstance(model_input, torch.Tensor) or model_input.ndim != 4:
            raise TypeError("Standard UNet variant masks require BCHW model input.")
        if not isinstance(transformer_options, dict):
            raise TypeError("Standard UNet variant options must be a dictionary.")
        self._validate_regions(region_indices, bank.region_count)
        layout = self._layout(transformer_options)
        height, width = int(model_input.shape[-2]), int(model_input.shape[-1])
        if layout is None:
            if (height, width) != (bank.canvas_height, bank.canvas_width):
                raise ValueError(
                    "Full standard-UNet output must match the regional mask canvas."
                )
            masks = (
                bank.conditioning_masks[list(region_indices)]
                .unsqueeze(1)
                .expand(
                    -1,
                    int(model_input.shape[0]),
                    -1,
                    -1,
                )
            )
        else:
            if layout.expanded_batch_size != int(model_input.shape[0]):
                raise ValueError("Spatial layout batch must match model output batch.")
            if any(
                (view.model_height, view.model_width) != (height, width)
                for view in layout.views
            ):
                raise ValueError("Spatial layout model shape must match model output.")
            views = tuple(
                self._projector.project_query_grid(
                    bank=bank,
                    layout=layout,
                    view_index=view_index,
                    query_height=height,
                    query_width=width,
                    form=RegionalMaskForm.CONDITIONING,
                    mode=RegionalMaskProjectionMode.CONTINUOUS_COVERAGE,
                )[list(region_indices)]
                for view_index in range(layout.view_count)
            )
            masks = torch.cat(
                tuple(
                    view.unsqueeze(1).expand(
                        -1,
                        layout.input_batch_size,
                        -1,
                        -1,
                    )
                    for view in views
                ),
                dim=1,
            )
        return masks.to(device=model_input.device, dtype=model_input.dtype).unsqueeze(2)

    @staticmethod
    def requires_base(
        bank: RegionalMaskBank,
        region_indices: tuple[int, ...],
    ) -> bool:
        """Report whether any canonical pixel lacks active variant ownership."""

        if not isinstance(bank, RegionalMaskBank):
            raise TypeError("Standard UNet base coverage requires a mask bank.")
        StandardUnetVariantMaskProjector._validate_regions(
            region_indices,
            bank.region_count,
        )
        coverage = bank.conditioning_masks[list(region_indices)].sum(dim=0)
        return bool((coverage < 1.0).any().item())

    @staticmethod
    def _layout(options: dict[str, object]) -> SpatialBatchLayout | None:
        """Return the optional authoritative spatial batch layout."""

        namespace = options.get(SIMPLE_SYRUP_TRANSFORMER_NAMESPACE)
        if namespace is None:
            return None
        if not isinstance(namespace, dict):
            raise TypeError("Standard UNet SimpleSyrup namespace must be a dictionary.")
        layout = namespace.get(SPATIAL_BATCH_LAYOUT_KEY)
        if layout is not None and not isinstance(layout, SpatialBatchLayout):
            raise TypeError("Standard UNet spatial layout has an invalid type.")
        return layout

    @staticmethod
    def _validate_regions(region_indices: tuple[int, ...], region_count: int) -> None:
        """Require unique canonical selected region indices."""

        if not isinstance(region_indices, tuple) or not region_indices:
            raise ValueError("Standard UNet variants require selected regions.")
        if region_indices != tuple(sorted(set(region_indices))):
            raise ValueError("Standard UNet variant regions must be unique and sorted.")
        if region_indices[0] < 0 or region_indices[-1] >= region_count:
            raise ValueError("Standard UNet variant region is outside the mask bank.")


STANDARD_UNET_VARIANT_MASK_PROJECTOR = StandardUnetVariantMaskProjector()
