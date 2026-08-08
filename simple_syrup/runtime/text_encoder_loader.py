# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load FLUX text encoders through ComfyUI's native CLIP loader."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

TEXT_ENCODER_DEVICES = ("default", "cpu")


class TextEncoderLoader:
    """Adapt ComfyUI text-encoder loading behind a typed boundary."""

    def __init__(self, folder_paths_module: ModuleType | None = None) -> None:
        """Create a loader with injectable ComfyUI folder paths."""

        self._folder_paths_module = folder_paths_module

    def load(
        self,
        paths: Sequence[Path],
        clip_type_name: str,
        device: str,
    ) -> object:
        """Load one or more encoder files for a named ComfyUI CLIP type."""

        if device not in TEXT_ENCODER_DEVICES:
            valid = ", ".join(TEXT_ENCODER_DEVICES)
            raise ValueError(f"text_encoder_device must be one of: {valid}.")
        if not paths:
            raise ValueError("At least one text encoder path is required.")

        comfy_sd: Any = importlib.import_module("comfy.sd")
        clip_type = getattr(comfy_sd.CLIPType, clip_type_name)
        model_options: dict[str, object] = {}
        if device == "cpu":
            cpu_device = torch.device("cpu")
            model_options = {
                "load_device": cpu_device,
                "offload_device": cpu_device,
            }
        return comfy_sd.load_clip(
            ckpt_paths=[str(path) for path in paths],
            embedding_directory=self._folder_paths().get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options,
        )

    def _folder_paths(self) -> ModuleType:
        """Return the configured ComfyUI folder paths module."""

        if self._folder_paths_module is not None:
            return self._folder_paths_module
        module: Any = importlib.import_module("folder_paths")
        if not isinstance(module, ModuleType):
            raise TypeError("folder_paths import did not return a module.")
        self._folder_paths_module = module
        return module
