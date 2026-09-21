# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Inspect ComfyUI-loaded model configuration for structural FLUX metadata."""

from __future__ import annotations

from typing import Protocol

from ..domain.flux_profiles import FluxModelProfile, classify_flux_profile
from .diffusion_model_metadata import (
    DiffusionModelMetadata,
    DiffusionModelMetadataInspector,
)


class FluxModelInspector:
    """Read ComfyUI's tensor-derived model configuration after model loading."""

    def __init__(
        self,
        metadata_inspector: DiffusionModelMetadataInspectorBoundary | None = None,
    ) -> None:
        """Create a FLUX classifier over shared metadata inspection."""

        self._metadata_inspector = (
            metadata_inspector or DiffusionModelMetadataInspector()
        )

    def inspect(self, model: object) -> FluxModelProfile | None:
        """Return a detected FLUX profile or None for unavailable metadata."""

        metadata = self._metadata_inspector.inspect(model)
        if metadata is None:
            return None
        return classify_flux_profile(
            metadata.image_model,
            metadata.context_input_dimension,
        )


class DiffusionModelMetadataInspectorBoundary(Protocol):
    """Expose normalized loaded diffusion-model metadata."""

    def inspect(self, model: object) -> DiffusionModelMetadata | None:
        """Return normalized metadata when available."""
