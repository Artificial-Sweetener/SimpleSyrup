# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for loading ViTMatte models."""

from __future__ import annotations

from typing import Any

from ..runtime.model_choices import ModelChoiceService, default_choice
from ..runtime.model_downloads import ComfyProgressReporter
from ..runtime.vitmatte_loader import ViTMatteLoaderService
from . import tooltips


class ViTMatteModelLoader:
    """Expose ViTMatte loading as a conventional model loader."""

    _service = ViTMatteLoaderService()
    _choices = ModelChoiceService()

    RETURN_TYPES = ("VITMATTE_MODEL",)
    RETURN_NAMES = ("vitmatte_model",)
    OUTPUT_TOOLTIPS = (tooltips.VITMATTE_MODEL_OUTPUT,)
    FUNCTION = "load_model"
    CATEGORY = "SimpleSyrup/Masking"
    DESCRIPTION = "Loads a ViTMatte model for Prompt SEGS w/ SAM edge refinement."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare deterministic ViTMatte loader inputs."""

        choices = cls._choices.vitmatte_choices()
        return {
            "required": {
                "vitmatte_model": (
                    choices,
                    {
                        "default": default_choice(
                            choices,
                            "vitmatte-small-composition-1k",
                        ),
                        "tooltip": tooltips.VITMATTE_MODEL_INPUT,
                    },
                ),
            }
        }

    def load_model(
        self,
        vitmatte_model: str,
    ) -> tuple[object]:
        """Load and return a ViTMatte-compatible model object."""

        self._choices.reject_sentinel(vitmatte_model)
        return (
            self._service.load_model(
                vitmatte_model=vitmatte_model,
                auto_download=True,
                progress=ComfyProgressReporter(),
            ),
        )
