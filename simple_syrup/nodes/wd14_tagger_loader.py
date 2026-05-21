# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for loading WD14 taggers."""

from __future__ import annotations

from typing import Any

from ..runtime.model_catalog import DEFAULT_WD14_TAGGER_MODEL
from ..runtime.model_choices import ModelChoiceService, default_choice
from ..runtime.model_downloads import ComfyProgressReporter
from ..runtime.wd14_tagger_loader import WD14TaggerLoaderService
from . import tooltips


class WD14TaggerLoader:
    """Expose WD14 tagger loading as a conventional ComfyUI model loader."""

    _service = WD14TaggerLoaderService()
    _choices = ModelChoiceService()

    RETURN_TYPES = ("WD14_TAGGER",)
    RETURN_NAMES = ("wd14_tagger",)
    OUTPUT_TOOLTIPS = (tooltips.WD14_TAGGER_OUTPUT,)
    FUNCTION = "load_model"
    CATEGORY = "SimpleSyrup/Tagging"
    DESCRIPTION = "Loads a WD14 tagger for compatible SimpleSyrup tagging workflows."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare deterministic WD14 tagger loader inputs."""

        choices = cls._choices.wd14_tagger_choices()
        return {
            "required": {
                "wd14_model": (
                    choices,
                    {
                        "default": default_choice(
                            choices,
                            DEFAULT_WD14_TAGGER_MODEL,
                        ),
                        "tooltip": tooltips.WD14_MODEL_INPUT,
                    },
                ),
            }
        }

    def load_model(self, wd14_model: str) -> tuple[object]:
        """Load and return a WD14 tagger model object."""

        self._choices.reject_sentinel(wd14_model)
        return (
            self._service.load_model(
                wd14_model=wd14_model,
                auto_download=True,
                progress=ComfyProgressReporter(),
            ),
        )
