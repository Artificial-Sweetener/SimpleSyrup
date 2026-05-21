# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for grounded SAM model metadata."""

from __future__ import annotations

from typing import Any

from ..runtime.model_catalog import grounding_dino_choices, sam_choices
from ..runtime.model_metadata import GroundedSAMModelMetadata
from . import tooltips


class GroundedSAMModelInfo:
    """Expose selected grounded SAM source and local path metadata."""

    _metadata = GroundedSAMModelMetadata()

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_info",)
    OUTPUT_TOOLTIPS = (
        "JSON metadata describing the selected SAM and GroundingDINO model sources.",
    )
    FUNCTION = "describe"
    CATEGORY = "SimpleSyrup/Masking"
    DESCRIPTION = "Returns JSON metadata for selected SAM and GroundingDINO models."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare deterministic model metadata inputs."""

        return {
            "required": {
                "sam_model": (
                    sam_choices(),
                    {
                        "default": "sam_hq_vit_b (379MB)",
                        "tooltip": tooltips.SAM_MODEL_INPUT,
                    },
                ),
                "grounding_dino_model": (
                    grounding_dino_choices(),
                    {
                        "default": "GroundingDINO_SwinT_OGC (694MB)",
                        "tooltip": tooltips.GROUNDING_DINO_MODEL_INPUT,
                    },
                ),
            }
        }

    def describe(self, sam_model: str, grounding_dino_model: str) -> tuple[str]:
        """Return JSON metadata for selected model entries."""

        return (self._metadata.describe_selection(sam_model, grounding_dino_model),)
