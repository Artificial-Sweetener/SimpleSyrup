# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for grounded SAM model metadata."""

from __future__ import annotations

from typing import Any

from ..runtime.model_choices import ModelChoiceService, default_choice
from ..runtime.model_metadata import GroundedSAMModelMetadata
from . import tooltips


class GroundedSAMModelInfo:
    """Expose selected grounded SAM source and local path metadata."""

    _metadata = GroundedSAMModelMetadata()
    _choices = ModelChoiceService()

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

        sam_model_choices = cls._choices.sam_choices()
        grounding_dino_model_choices = cls._choices.grounding_dino_choices()
        return {
            "required": {
                "sam_model": (
                    sam_model_choices,
                    {
                        "default": default_choice(
                            sam_model_choices,
                            "sam_hq_vit_b (379MB)",
                        ),
                        "tooltip": tooltips.SAM_MODEL_INPUT,
                    },
                ),
                "grounding_dino_model": (
                    grounding_dino_model_choices,
                    {
                        "default": default_choice(
                            grounding_dino_model_choices,
                            "GroundingDINO_SwinT_OGC (694MB)",
                        ),
                        "tooltip": tooltips.GROUNDING_DINO_MODEL_INPUT,
                    },
                ),
            }
        }

    def describe(self, sam_model: str, grounding_dino_model: str) -> tuple[str]:
        """Return JSON metadata for selected model entries."""

        self._choices.reject_sentinel(sam_model)
        self._choices.reject_sentinel(grounding_dino_model)
        return (self._metadata.describe_selection(sam_model, grounding_dino_model),)
