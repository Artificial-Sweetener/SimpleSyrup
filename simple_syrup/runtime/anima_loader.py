# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load Anima diffusion, text encoder, and VAE models for ComfyUI."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import torch

from .auto_model_resolver import AutoModelResolution, AutoModelResolver
from .model_catalog import ANIMA_QWEN_TEXT_ENCODER, ANIMA_QWEN_VAE, AutoModelArtifact
from .model_downloads import ProgressReporter
from .vae_loader import VaeLoaderService, load_vae_path

AUTO_CHOICE = "auto"
DIFFUSION_WEIGHT_DTYPES = (
    "default",
    "fp8_e4m3fn",
    "fp8_e4m3fn_fast",
    "fp8_e5m2",
)
CLIP_DEVICES = ("default", "cpu")
DEFAULT_CLIP_TYPE = "stable_diffusion"


class AnimaLoaderService:
    """Resolve and load Anima diffusion, text encoder, and VAE models."""

    def __init__(
        self,
        resolver: AutoModelResolverBoundary | None = None,
        folder_paths_module: ModuleType | None = None,
    ) -> None:
        """Create a loader with injectable auto-resolution boundaries."""

        self._folder_paths_module = folder_paths_module
        self._resolver = resolver or AutoModelResolver(
            folder_paths_module=folder_paths_module
        )
        self._vae_loader = VaeLoaderService(folder_paths_module)

    def load_models(
        self,
        diffusion_model: str,
        diffusion_weight_dtype: str,
        text_encoder: str,
        text_encoder_device: str,
        vae: str,
        progress: ProgressReporter | None = None,
    ) -> tuple[object, object, object]:
        """Return ComfyUI MODEL, CLIP, and VAE objects."""

        return (
            self._load_diffusion_model(diffusion_model, diffusion_weight_dtype),
            self._load_clip(
                text_encoder,
                text_encoder_device,
                progress,
            ),
            self._load_vae(vae, progress),
        )

    def _load_diffusion_model(
        self,
        diffusion_model: str,
        diffusion_weight_dtype: str,
    ) -> object:
        """Load a diffusion model using ComfyUI's diffusion model loader policy."""

        if diffusion_weight_dtype not in DIFFUSION_WEIGHT_DTYPES:
            valid = ", ".join(DIFFUSION_WEIGHT_DTYPES)
            raise ValueError(f"diffusion_weight_dtype must be one of: {valid}.")

        model_options: dict[str, object] = {}
        if diffusion_weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif diffusion_weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif diffusion_weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        folder_paths = self._folder_paths()
        unet_path = folder_paths.get_full_path_or_raise(
            "diffusion_models",
            diffusion_model,
        )
        comfy_sd = _comfy_sd()
        return comfy_sd.load_diffusion_model(
            unet_path,
            model_options=model_options,
        )

    def _load_clip(
        self,
        text_encoder: str,
        text_encoder_device: str,
        progress: ProgressReporter | None,
    ) -> object:
        """Load a CLIP/text encoder using ComfyUI's CLIP loader policy."""

        if text_encoder_device not in CLIP_DEVICES:
            valid = ", ".join(CLIP_DEVICES)
            raise ValueError(f"text_encoder_device must be one of: {valid}.")

        if text_encoder == AUTO_CHOICE:
            clip_path = self._resolver.resolve(ANIMA_QWEN_TEXT_ENCODER, progress).path
        else:
            clip_path = Path(
                str(
                    self._folder_paths().get_full_path_or_raise(
                        "text_encoders",
                        text_encoder,
                    )
                )
            )

        comfy_sd = _comfy_sd()
        clip_type = getattr(
            comfy_sd.CLIPType,
            DEFAULT_CLIP_TYPE.upper(),
            comfy_sd.CLIPType.STABLE_DIFFUSION,
        )
        model_options: dict[str, object] = {}
        if text_encoder_device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = (
                torch.device("cpu")
            )

        return comfy_sd.load_clip(
            ckpt_paths=[str(clip_path)],
            embedding_directory=self._folder_paths().get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options,
        )

    def _load_vae(
        self,
        vae: str,
        progress: ProgressReporter | None,
    ) -> object:
        """Load a VAE using ComfyUI's VAE loader policy."""

        if vae == AUTO_CHOICE:
            vae_path = self._resolver.resolve(ANIMA_QWEN_VAE, progress).path
            return load_vae_path(vae_path)

        return self._vae_loader.load_vae(vae)

    def _folder_paths(self) -> ModuleType:
        """Return the ComfyUI folder_paths module."""

        if self._folder_paths_module is not None:
            return self._folder_paths_module
        module: Any = importlib.import_module("folder_paths")
        if not isinstance(module, ModuleType):
            raise TypeError("folder_paths import did not return a module.")
        self._folder_paths_module = module
        return module


class AutoModelResolverBoundary(Protocol):
    """Resolver interface required by the Anima loader service."""

    def resolve(
        self,
        artifact: AutoModelArtifact,
        progress: ProgressReporter | None = None,
    ) -> AutoModelResolution:
        """Resolve one automatic model artifact."""


def _folder_paths() -> ModuleType:
    """Import ComfyUI folder paths lazily."""

    module: Any = importlib.import_module("folder_paths")
    if not isinstance(module, ModuleType):
        raise TypeError("folder_paths import did not return a module.")
    return module


def _comfy_sd() -> Any:
    """Import ComfyUI's stable diffusion loading module lazily."""

    return importlib.import_module("comfy.sd")
