# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for simple Anima model loading."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from ..runtime.anima_loader import (
    AUTO_CHOICE,
    CLIP_DEVICES,
    DIFFUSION_WEIGHT_DTYPES,
    AnimaLoaderService,
)
from ..runtime.model_downloads import ComfyProgressReporter
from ..runtime.vae_loader import vae_choices
from . import tooltips


class SimpleLoadAnima:
    """Expose Anima diffusion, text encoder, and VAE loading as one node."""

    _service = AnimaLoaderService()

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    OUTPUT_TOOLTIPS = (
        tooltips.MODEL_OUTPUT,
        tooltips.CLIP_OUTPUT,
        tooltips.VAE_OUTPUT,
    )
    FUNCTION = "load_models"
    CATEGORY = "SimpleSyrup/Loaders"
    DESCRIPTION = "Loads Anima model components with auto-resolved Qwen assets."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare deterministic Simple Load Anima inputs."""

        folder_paths = _folder_paths()
        return {
            "required": {
                "diffusion_model": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {
                        "tooltip": (
                            "Anima diffusion model file used as the main image "
                            "generation model."
                        )
                    },
                ),
                "diffusion_weight_dtype": (
                    list(DIFFUSION_WEIGHT_DTYPES),
                    {
                        "default": "default",
                        "advanced": True,
                        "tooltip": (
                            "Weight precision for Anima. Lower precision can reduce "
                            "memory use but may slightly change results."
                        ),
                    },
                ),
                "text_encoder": (
                    _choices_with_auto(folder_paths.get_filename_list("text_encoders")),
                    {
                        "default": AUTO_CHOICE,
                        "tooltip": (
                            "Qwen text encoder used for Anima prompt understanding. "
                            "Auto selects the expected model."
                        ),
                    },
                ),
                "text_encoder_device": (
                    list(CLIP_DEVICES),
                    {
                        "default": "default",
                        "advanced": True,
                        "tooltip": (
                            "Device for the text encoder. Auto follows the normal "
                            "ComfyUI placement."
                        ),
                    },
                ),
                "vae": (
                    _choices_with_auto(vae_choices(folder_paths)),
                    {
                        "default": AUTO_CHOICE,
                        "tooltip": (
                            "VAE used to decode Anima latents. Auto selects the "
                            "expected Qwen image VAE."
                        ),
                    },
                ),
            }
        }

    def load_models(
        self,
        diffusion_model: str,
        diffusion_weight_dtype: str,
        text_encoder: str,
        text_encoder_device: str,
        vae: str,
    ) -> tuple[object, object, object]:
        """Load and return Anima's MODEL, CLIP, and VAE objects."""

        return self._service.load_models(
            diffusion_model=diffusion_model,
            diffusion_weight_dtype=diffusion_weight_dtype,
            text_encoder=text_encoder,
            text_encoder_device=text_encoder_device,
            vae=vae,
            progress=ComfyProgressReporter(),
        )


def _choices_with_auto(choices: list[str]) -> list[str]:
    """Return choices with the automatic selection first and deduplicated."""

    return [AUTO_CHOICE, *(choice for choice in choices if choice != AUTO_CHOICE)]


def _folder_paths() -> ModuleType:
    """Import ComfyUI folder paths lazily."""

    module: Any = importlib.import_module("folder_paths")
    if not isinstance(module, ModuleType):
        raise TypeError("folder_paths import did not return a module.")
    return module
