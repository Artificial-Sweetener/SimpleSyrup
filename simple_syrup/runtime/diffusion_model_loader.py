# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load standalone diffusion models through ComfyUI's native policy."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import torch

DIFFUSION_WEIGHT_DTYPES = (
    "default",
    "fp8_e4m3fn",
    "fp8_e4m3fn_fast",
    "fp8_e5m2",
)


class DiffusionModelLoader:
    """Adapt ComfyUI's standalone diffusion loader behind a typed boundary."""

    def __init__(self, folder_paths_module: ModuleType | None = None) -> None:
        """Create a loader with injectable ComfyUI folder paths."""

        self._folder_paths_module = folder_paths_module

    def load(self, diffusion_model: str, weight_dtype: str) -> object:
        """Load one diffusion model using a validated weight dtype."""

        model_options = diffusion_model_options(weight_dtype)
        model_path = self._folder_paths().get_full_path_or_raise(
            "diffusion_models",
            diffusion_model,
        )
        comfy_sd: Any = importlib.import_module("comfy.sd")
        return comfy_sd.load_diffusion_model(
            model_path,
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


def diffusion_model_options(weight_dtype: str) -> dict[str, object]:
    """Return ComfyUI model options for a supported diffusion weight dtype."""

    if weight_dtype not in DIFFUSION_WEIGHT_DTYPES:
        valid = ", ".join(DIFFUSION_WEIGHT_DTYPES)
        raise ValueError(f"diffusion_weight_dtype must be one of: {valid}.")
    if weight_dtype == "fp8_e4m3fn":
        return {"dtype": torch.float8_e4m3fn}
    if weight_dtype == "fp8_e4m3fn_fast":
        return {"dtype": torch.float8_e4m3fn, "fp8_optimizations": True}
    if weight_dtype == "fp8_e5m2":
        return {"dtype": torch.float8_e5m2}
    return {}
