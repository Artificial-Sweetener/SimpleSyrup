# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Inspect tensor-derived metadata exposed by loaded ComfyUI diffusion models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..shared.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class DiffusionModelMetadata:
    """Describe architecture fields relevant to component selection."""

    image_model: str | None
    context_input_dimension: int | None


@runtime_checkable
class ModelPatcherBoundary(Protocol):
    """Expose the loaded model objects required for architecture inspection."""

    def get_model_object(self, name: str) -> object:
        """Return a named object owned by ComfyUI's model patcher."""


class DiffusionModelMetadataInspector:
    """Read normalized architecture metadata from a loaded model patcher."""

    def inspect(self, model: object) -> DiffusionModelMetadata | None:
        """Return narrowed model metadata or None when it is unavailable."""

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
        return DiffusionModelMetadata(
            image_model=(
                image_model_value if isinstance(image_model_value, str) else None
            ),
            context_input_dimension=(
                context_dimension_value
                if isinstance(context_dimension_value, int)
                and not isinstance(context_dimension_value, bool)
                else None
            ),
        )
