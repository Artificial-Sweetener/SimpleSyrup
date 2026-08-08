# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Inspect ComfyUI-loaded model configuration for structural FLUX metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from ..domain.flux_profiles import FluxModelProfile, classify_flux_profile
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)


@runtime_checkable
class ModelPatcherBoundary(Protocol):
    """Expose the loaded model objects required for architecture inspection."""

    def get_model_object(self, name: str) -> object:
        """Return a named object owned by ComfyUI's model patcher."""


class FluxModelInspector:
    """Read ComfyUI's tensor-derived model configuration after model loading."""

    def inspect(self, model: object) -> FluxModelProfile | None:
        """Return a detected FLUX profile or None for unavailable metadata."""

        if not isinstance(model, ModelPatcherBoundary):
            return None
        try:
            model_config = model.get_model_object("model_config")
        except (AttributeError, KeyError, TypeError, ValueError):
            LOGGER.warning(
                "loaded model does not expose inspectable model configuration"
            )
            return None

        unet_config = getattr(model_config, "unet_config", None)
        if not isinstance(unet_config, Mapping):
            return None
        image_model_value = unet_config.get("image_model")
        context_dimension_value = unet_config.get("context_in_dim")
        image_model = image_model_value if isinstance(image_model_value, str) else None
        context_dimension = (
            context_dimension_value
            if isinstance(context_dimension_value, int)
            and not isinstance(context_dimension_value, bool)
            else None
        )
        return classify_flux_profile(image_model, context_dimension)
