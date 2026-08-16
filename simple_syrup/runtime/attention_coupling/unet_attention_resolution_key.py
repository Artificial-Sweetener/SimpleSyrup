# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Identify reusable standard-UNet attention resolution state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...domain.spatial_views import SpatialBatchLayout
from .unet_attention_geometry import StandardUnetAttentionGeometry


@dataclass(frozen=True, slots=True, eq=False)
class StandardUnetAttentionResolutionKey:
    """Identify every call-local value that changes projected execution."""

    input_batch_size: int
    query_height: int
    query_width: int
    original_height: int
    original_width: int
    device: torch.device
    dtype: torch.dtype
    published_layout: SpatialBatchLayout | None

    @classmethod
    def from_geometry(
        cls,
        geometry: StandardUnetAttentionGeometry,
        query: torch.Tensor,
    ) -> StandardUnetAttentionResolutionKey:
        """Build a key from validated geometry and query residency."""

        if not isinstance(geometry, StandardUnetAttentionGeometry):
            raise TypeError("UNet resolution key requires validated geometry.")
        if not isinstance(query, torch.Tensor):
            raise TypeError("UNet resolution key requires a query tensor.")
        return cls(
            geometry.query.input_batch_size,
            geometry.query.query_height,
            geometry.query.query_width,
            geometry.original_height,
            geometry.original_width,
            query.device,
            query.dtype,
            geometry.query.spatial_layout,
        )

    def __hash__(self) -> int:
        """Hash exact scalar residency plus published layout identity."""

        return hash(
            (
                self.input_batch_size,
                self.query_height,
                self.query_width,
                self.original_height,
                self.original_width,
                self.device,
                self.dtype,
                id(self.published_layout),
            )
        )

    def __eq__(self, other: object) -> bool:
        """Compare every scalar and require exact published layout identity."""

        return (
            isinstance(other, StandardUnetAttentionResolutionKey)
            and self.input_batch_size == other.input_batch_size
            and self.query_height == other.query_height
            and self.query_width == other.query_width
            and self.original_height == other.original_height
            and self.original_width == other.original_width
            and self.device == other.device
            and self.dtype == other.dtype
            and self.published_layout is other.published_layout
        )
