# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for loading GroundingDINO models."""

from __future__ import annotations

from typing import Any

from ..runtime.grounding_dino_loader import (
    TEXT_ENCODER_AUTO,
    TEXT_ENCODER_CHOICES,
    GroundingDINOLoaderService,
)
from ..runtime.model_choices import ModelChoiceService, default_choice
from ..runtime.model_downloads import ComfyProgressReporter
from . import tooltips


class GroundingDINOModelLoader:
    """Expose GroundingDINO loading with explicit text encoder selection."""

    _service = GroundingDINOLoaderService()
    _choices = ModelChoiceService()

    RETURN_TYPES = ("GROUNDING_DINO_MODEL",)
    RETURN_NAMES = ("grounding_dino_model",)
    OUTPUT_TOOLTIPS = (tooltips.GROUNDING_DINO_MODEL_OUTPUT,)
    FUNCTION = "load_model"
    CATEGORY = "SimpleSyrup/Masking"
    DESCRIPTION = "Loads GroundingDINO and its explicit BERT text encoder."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare deterministic GroundingDINO loader inputs."""

        choices = cls._choices.grounding_dino_choices()
        return {
            "required": {
                "grounding_dino_model": (
                    choices,
                    {
                        "default": default_choice(
                            choices,
                            "GroundingDINO_SwinT_OGC (694MB)",
                        ),
                        "tooltip": tooltips.GROUNDING_DINO_MODEL_INPUT,
                    },
                ),
                "text_encoder": (
                    list(TEXT_ENCODER_CHOICES),
                    {
                        "default": TEXT_ENCODER_AUTO,
                        "tooltip": tooltips.GROUNDING_DINO_TEXT_ENCODER_INPUT,
                    },
                ),
            }
        }

    def load_model(
        self,
        grounding_dino_model: str,
        text_encoder: str,
    ) -> tuple[object]:
        """Load and return a GroundingDINO-compatible model object."""

        self._choices.reject_sentinel(grounding_dino_model)
        return (
            self._service.load_model(
                grounding_dino_model=grounding_dino_model,
                text_encoder=text_encoder,
                auto_download=True,
                progress=ComfyProgressReporter(),
            ),
        )
