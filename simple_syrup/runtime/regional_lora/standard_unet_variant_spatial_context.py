# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve standard-UNet variant spatial execution context."""

from __future__ import annotations

from ...domain.spatial_views import SpatialBatchLayout
from ..spatial_model_arguments import (
    SIMPLE_SYRUP_TRANSFORMER_NAMESPACE,
    SPATIAL_BATCH_LAYOUT_KEY,
)


class StandardUnetVariantSpatialContext:
    """Own optional spatial-layout parsing and execution-mode identity."""

    def layout(
        self,
        transformer_options: dict[str, object],
    ) -> SpatialBatchLayout | None:
        """Return the authoritative optional layout or fail closed."""

        if not isinstance(transformer_options, dict):
            raise TypeError("Standard UNet transformer options must be a dictionary.")
        namespace = transformer_options.get(SIMPLE_SYRUP_TRANSFORMER_NAMESPACE)
        if namespace is None:
            return None
        if not isinstance(namespace, dict):
            raise TypeError("Standard UNet spatial namespace must be a dictionary.")
        layout = namespace.get(SPATIAL_BATCH_LAYOUT_KEY)
        if layout is not None and not isinstance(layout, SpatialBatchLayout):
            raise TypeError("Standard UNet spatial layout has an invalid type.")
        return layout

    def modes(self, transformer_options: dict[str, object]) -> tuple[str, ...]:
        """Return the exact model-call spatial kind for durable evidence."""

        layout = self.layout(transformer_options)
        if layout is None:
            return ("full",)
        return (layout.views[0].kind.value,)


STANDARD_UNET_VARIANT_SPATIAL_CONTEXT = StandardUnetVariantSpatialContext()
