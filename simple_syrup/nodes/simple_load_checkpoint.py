# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for checkpoint loading with optional VAE override."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from ..runtime.checkpoint_loader import (
    CLIP_SKIP_DEFAULT,
    USE_CHECKPOINT_VAE_CHOICE,
    CheckpointLoaderService,
)
from ..runtime.vae_loader import vae_choices
from . import tooltips


class SimpleLoadCheckpoint:
    """Expose checkpoint loading with an optional external VAE override."""

    _service = CheckpointLoaderService()

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    OUTPUT_TOOLTIPS = (
        tooltips.MODEL_OUTPUT,
        tooltips.CLIP_OUTPUT,
        tooltips.VAE_OUTPUT,
    )
    FUNCTION = "load_checkpoint"
    CATEGORY = "SimpleSyrup/Loaders"
    DESCRIPTION = "Loads a checkpoint and optionally replaces its VAE."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare deterministic Simple Load Checkpoint inputs."""

        folder_paths = _folder_paths()
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": tooltips.CHECKPOINT_MODEL_INPUT},
                ),
                "vae_name": (
                    _choices_with_checkpoint_vae(vae_choices(folder_paths)),
                    {
                        "default": USE_CHECKPOINT_VAE_CHOICE,
                        "tooltip": tooltips.CHECKPOINT_VAE_INPUT,
                    },
                ),
                "clip_skip": (
                    "BOOLEAN",
                    {
                        "default": CLIP_SKIP_DEFAULT,
                        "tooltip": tooltips.CLIP_SKIP_INPUT,
                    },
                ),
            }
        }

    def load_checkpoint(
        self,
        ckpt_name: str,
        vae_name: str,
        clip_skip: bool,
    ) -> tuple[object, object, object]:
        """Load checkpoint MODEL and CLIP with the selected VAE."""

        return self._service.load_checkpoint(
            ckpt_name=ckpt_name,
            vae_name=vae_name,
            clip_skip=clip_skip,
        )


def _choices_with_checkpoint_vae(choices: list[str]) -> list[str]:
    """Return choices with the checkpoint VAE selection first and deduplicated."""

    return [
        USE_CHECKPOINT_VAE_CHOICE,
        *(choice for choice in choices if choice != USE_CHECKPOINT_VAE_CHOICE),
    ]


def _folder_paths() -> ModuleType:
    """Import ComfyUI folder paths lazily."""

    module: Any = importlib.import_module("folder_paths")
    if not isinstance(module, ModuleType):
        raise TypeError("folder_paths import did not return a module.")
    return module
