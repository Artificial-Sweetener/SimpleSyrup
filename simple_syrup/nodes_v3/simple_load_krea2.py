# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose cohesive Krea 2 component loading through Comfy's v3 API."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar

from ..nodes import tooltips
from ..runtime.auto_model_choices import automatic_component_choices
from ..runtime.diffusion_model_loader import DIFFUSION_WEIGHT_DTYPES
from ..runtime.krea2_artifacts import (
    KREA2_AUTO_TEXT_ENCODER,
    KREA2_QWEN3_VL_4B_BF16,
    KREA2_QWEN3_VL_4B_FP8,
)
from ..runtime.model_downloads import ComfyProgressReporter
from ..runtime.qwen_artifacts import QWEN_IMAGE_VAE
from ..runtime.text_encoder_loader import TEXT_ENCODER_DEVICES
from ..runtime.vae_loader import vae_choices
from ..services.krea2_loader_service import AUTO_CHOICE, Krea2LoaderService

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


class SimpleLoadKrea2V3(_ComfyNodeBase):
    """Load a Krea 2 diffusion model with its Qwen encoder and image VAE."""

    _service = Krea2LoaderService()

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the Krea 2 loader schema and downloadable component choices."""

        folder_paths = _folder_paths()
        return _comfy_io.Schema(
            node_id="SimpleSyrup.SimpleLoadKrea2",
            display_name="Simple Load Krea 2",
            category="SimpleSyrup/Loaders",
            description=(
                "Loads Krea 2 with its Qwen3-VL 4B encoder and Qwen Image VAE; "
                "automatic components are downloaded from checksum-pinned "
                "Hugging Face files."
            ),
            search_aliases=["krea", "krea 2", "k2", "load krea"],
            inputs=[
                _comfy_io.Combo.Input(
                    "diffusion_model",
                    options=list(folder_paths.get_filename_list("diffusion_models")),
                    tooltip=(
                        "Krea 2 Raw or Turbo diffusion model to load. This node "
                        "validates the architecture and never downloads this file."
                    ),
                ),
                _comfy_io.Combo.Input(
                    "diffusion_weight_dtype",
                    options=list(DIFFUSION_WEIGHT_DTYPES),
                    default="default",
                    advanced=True,
                    tooltip=(
                        "Load-time diffusion precision; default preserves the "
                        "selected file's stored BF16, FP8, INT8, MXFP8, or NVFP4 "
                        "format."
                    ),
                ),
                _comfy_io.Combo.Input(
                    "text_encoder",
                    options=automatic_component_choices(
                        installed=list(folder_paths.get_filename_list("text_encoders")),
                        artifacts=(
                            KREA2_QWEN3_VL_4B_FP8,
                            KREA2_QWEN3_VL_4B_BF16,
                        ),
                        leading_choices=(
                            KREA2_AUTO_TEXT_ENCODER,
                            KREA2_QWEN3_VL_4B_FP8.filename,
                            KREA2_QWEN3_VL_4B_BF16.filename,
                        ),
                        folder_paths_module=folder_paths,
                    ),
                    default=KREA2_AUTO_TEXT_ENCODER,
                    advanced=True,
                    tooltip=(
                        "Qwen3-VL 4B encoder loaded with Krea 2's required 12-layer "
                        "conditioning. Auto uses FP8; selecting official FP8 or BF16 "
                        "downloads that checksum-pinned file when missing."
                    ),
                ),
                _comfy_io.Combo.Input(
                    "text_encoder_device",
                    options=list(TEXT_ENCODER_DEVICES),
                    default="default",
                    advanced=True,
                    tooltip=(
                        "Device for Qwen3-VL; CPU saves GPU memory but makes prompt "
                        "encoding slower."
                    ),
                ),
                _comfy_io.Combo.Input(
                    "vae",
                    options=automatic_component_choices(
                        installed=vae_choices(folder_paths),
                        artifacts=(QWEN_IMAGE_VAE,),
                        leading_choices=(AUTO_CHOICE,),
                        folder_paths_module=folder_paths,
                    ),
                    default=AUTO_CHOICE,
                    advanced=True,
                    tooltip=(
                        "VAE used to decode Krea 2 latents. Auto finds or downloads "
                        "the checksum-pinned Qwen Image VAE with visible progress."
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
        text_encoder: str,
        text_encoder_device: str,
        vae: str,
    ) -> tuple[object, object, object]:
        """Load and return validated Krea 2 MODEL, CLIP, and VAE objects."""

        return cls._service.load_models(
            diffusion_model=diffusion_model,
            diffusion_weight_dtype=diffusion_weight_dtype,
            text_encoder=text_encoder,
            text_encoder_device=text_encoder_device,
            vae=vae,
            progress=ComfyProgressReporter(),
        )


def _folder_paths() -> ModuleType:
    """Import ComfyUI folder paths lazily for schema declaration."""

    module: Any = importlib.import_module("folder_paths")
    if not isinstance(module, ModuleType):
        raise TypeError("folder_paths import did not return a module.")
    return module
