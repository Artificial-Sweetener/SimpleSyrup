# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for loading SAM models."""

from __future__ import annotations

from typing import Any

from ..runtime.model_choices import ModelChoiceService, default_choice
from ..runtime.model_downloads import ComfyProgressReporter
from ..runtime.progress import create_comfy_phase_progress
from ..runtime.sam_loader import SAMLoaderService
from . import tooltips


class SAMModelLoader:
    """Expose SAM loading as a conventional ComfyUI model loader."""

    _service = SAMLoaderService()
    _choices = ModelChoiceService()

    RETURN_TYPES = ("SAM_MODEL",)
    RETURN_NAMES = ("sam_model",)
    OUTPUT_TOOLTIPS = (tooltips.SAM_MODEL_OUTPUT,)
    FUNCTION = "load_model"
    CATEGORY = "SimpleSyrup/Masking"
    DESCRIPTION = "Loads a SAM model for Prompt SEGS w/ SAM and compatible nodes."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare deterministic SAM loader inputs."""

        choices = cls._choices.sam_choices()
        return {
            "required": {
                "sam_model": (
                    choices,
                    {
                        "default": default_choice(choices, "sam_vit_b (375MB)"),
                        "tooltip": tooltips.SAM_MODEL_INPUT,
                    },
                ),
            }
        }

    def load_model(
        self,
        sam_model: str,
    ) -> tuple[object]:
        """Load and return a SAM-compatible model object."""

        self._choices.reject_sentinel(sam_model)
        return (
            self._service.load_model(
                sam_model=sam_model,
                auto_download=True,
                progress=ComfyProgressReporter(),
                phase_progress=create_comfy_phase_progress(
                    operation="sam_model_load",
                    subject=sam_model,
                    total_phases=4,
                ),
            ),
        )
