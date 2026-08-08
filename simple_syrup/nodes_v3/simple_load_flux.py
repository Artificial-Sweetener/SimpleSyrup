# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose cohesive FLUX.1 model loading through Comfy's v3 API."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar

from ..nodes import tooltips
from ..runtime.diffusion_model_loader import DIFFUSION_WEIGHT_DTYPES
from ..runtime.model_downloads import ComfyProgressReporter
from ..runtime.text_encoder_loader import TEXT_ENCODER_DEVICES
from ..runtime.vae_loader import vae_choices
from ..services.flux_loader_components import AUTO_CHOICE
from ..services.flux_loader_service import FluxLoaderService

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = importlib.import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = (
    None if TYPE_CHECKING else importlib.import_module("comfy_api.latest").io
)


class SimpleLoadFluxV3(_ComfyNodeBase):
    """Load FLUX.1 diffusion, dual text encoders, and VAE as one node."""

    _service = FluxLoaderService()

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the separate FLUX.1 loader schema."""

        folder_paths = _folder_paths()
        text_encoder_choices = _choices_with_auto(
            list(folder_paths.get_filename_list("text_encoders"))
        )
        return _comfy_io.Schema(
            node_id="SimpleSyrup.SimpleLoadFlux",
            display_name="Simple Load FLUX",
            category="SimpleSyrup/Loaders",
            description=(
                "Loads FLUX.1 with CLIP-L, T5-XXL, and its VAE; automatic "
                "components are downloaded from checksum-pinned Hugging Face files."
            ),
            search_aliases=["flux", "flux 1", "load flux"],
            inputs=[
                _comfy_io.Combo.Input(
                    "diffusion_model",
                    options=list(folder_paths.get_filename_list("diffusion_models")),
                    tooltip=(
                        "FLUX.1 diffusion model to load. This node never downloads "
                        "the diffusion model."
                    ),
                ),
                _comfy_io.Combo.Input(
                    "diffusion_weight_dtype",
                    options=list(DIFFUSION_WEIGHT_DTYPES),
                    default="default",
                    advanced=True,
                    tooltip=(
                        "Weight precision for the diffusion model; FP8 uses less "
                        "memory but can slightly change results."
                    ),
                ),
                _comfy_io.Combo.Input(
                    "clip_l",
                    options=text_encoder_choices,
                    default=AUTO_CHOICE,
                    advanced=True,
                    tooltip=(
                        "CLIP-L encoder for FLUX.1 prompts. Auto finds the exact "
                        "file or downloads it while Comfy shows node progress."
                    ),
                ),
                _comfy_io.Combo.Input(
                    "t5_xxl",
                    options=text_encoder_choices,
                    default=AUTO_CHOICE,
                    advanced=True,
                    tooltip=(
                        "T5-XXL encoder for FLUX.1 prompts. Auto may download the "
                        "large FP16 file and reports progress in Comfy."
                    ),
                ),
                _comfy_io.Combo.Input(
                    "text_encoder_device",
                    options=list(TEXT_ENCODER_DEVICES),
                    default="default",
                    advanced=True,
                    tooltip=(
                        "Device for both text encoders; CPU saves GPU memory but "
                        "makes prompt encoding slower."
                    ),
                ),
                _comfy_io.Combo.Input(
                    "vae",
                    options=_choices_with_auto(vae_choices(folder_paths)),
                    default=AUTO_CHOICE,
                    advanced=True,
                    tooltip=(
                        "VAE used to decode FLUX.1 latents. Auto finds or downloads "
                        "the checksum-pinned FLUX VAE with visible node progress."
                    ),
                ),
            ],
            outputs=[
                _comfy_io.Model.Output("model", tooltip=tooltips.MODEL_OUTPUT),
                _comfy_io.Clip.Output("clip", tooltip=tooltips.CLIP_OUTPUT),
                _comfy_io.Vae.Output("vae", tooltip=tooltips.VAE_OUTPUT),
            ],
        )

    @classmethod
    def execute(
        cls,
        diffusion_model: str,
        diffusion_weight_dtype: str,
        clip_l: str,
        t5_xxl: str,
        text_encoder_device: str,
        vae: str,
    ) -> tuple[object, object, object]:
        """Load and return FLUX.1 MODEL, CLIP, and VAE objects."""

        return cls._service.load_models(
            diffusion_model=diffusion_model,
            diffusion_weight_dtype=diffusion_weight_dtype,
            clip_l=clip_l,
            t5_xxl=t5_xxl,
            text_encoder_device=text_encoder_device,
            vae=vae,
            progress=ComfyProgressReporter(),
        )


def _choices_with_auto(choices: list[str]) -> list[str]:
    """Return deduplicated choices with automatic selection first."""

    return [AUTO_CHOICE, *(choice for choice in choices if choice != AUTO_CHOICE)]


def _folder_paths() -> ModuleType:
    """Import ComfyUI folder paths lazily for schema declaration."""

    module: Any = importlib.import_module("folder_paths")
    if not isinstance(module, ModuleType):
        raise TypeError("folder_paths import did not return a module.")
    return module
