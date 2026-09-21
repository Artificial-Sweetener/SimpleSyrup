# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load and validate the diffusion, text encoder, and VAE for Krea 2."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from ..runtime.auto_model_artifact import AutoModelArtifact
from ..runtime.auto_model_resolver import AutoModelResolution, AutoModelResolver
from ..runtime.clip_type_support import ComfyClipTypeSupport
from ..runtime.diffusion_model_loader import DiffusionModelLoader
from ..runtime.diffusion_model_metadata import (
    DiffusionModelMetadata,
    DiffusionModelMetadataInspector,
)
from ..runtime.krea2_artifacts import (
    KREA2_AUTO_TEXT_ENCODER,
    KREA2_QWEN3_VL_4B_FP8,
    KREA2_TEXT_ENCODER_ARTIFACTS,
)
from ..runtime.model_downloads import ProgressReporter
from ..runtime.qwen_artifacts import QWEN_IMAGE_VAE
from ..runtime.text_encoder_loader import TextEncoderLoader
from ..runtime.vae_loader import VaeLoaderService, load_vae_path

AUTO_CHOICE = "auto"
KREA2_CLIP_TYPE = "KREA2"
KREA2_IMAGE_MODEL = "krea2"


class Krea2LoaderService:
    """Orchestrate structurally validated Krea 2 component loading."""

    def __init__(
        self,
        diffusion_loader: DiffusionModelLoaderBoundary | None = None,
        text_encoder_loader: TextEncoderLoaderBoundary | None = None,
        model_inspector: DiffusionModelInspectorBoundary | None = None,
        clip_type_support: ClipTypeSupportBoundary | None = None,
        resolver: AutoModelResolverBoundary | None = None,
        vae_loader: VaeLoaderBoundary | None = None,
        folder_paths_module: ModuleType | None = None,
    ) -> None:
        """Create a loader with injectable host and artifact boundaries."""

        self._folder_paths_module = folder_paths_module
        self._diffusion_loader = diffusion_loader or DiffusionModelLoader(
            folder_paths_module
        )
        self._text_encoder_loader = text_encoder_loader or TextEncoderLoader(
            folder_paths_module
        )
        self._model_inspector = model_inspector or DiffusionModelMetadataInspector()
        self._clip_type_support = clip_type_support or ComfyClipTypeSupport()
        self._resolver = resolver or AutoModelResolver(
            folder_paths_module=folder_paths_module
        )
        self._vae_loader = vae_loader or VaeLoaderService(folder_paths_module)

    def load_models(
        self,
        diffusion_model: str,
        diffusion_weight_dtype: str,
        text_encoder: str,
        text_encoder_device: str,
        vae: str,
        progress: ProgressReporter | None = None,
    ) -> tuple[object, object, object]:
        """Return a validated Krea 2 MODEL, CLIP, and VAE tuple."""

        model = self._diffusion_loader.load(
            diffusion_model,
            diffusion_weight_dtype,
        )
        self._require_krea2_model(model)
        self._clip_type_support.require(KREA2_CLIP_TYPE)

        encoder_path = self._resolve_text_encoder(text_encoder, progress)
        clip = self._text_encoder_loader.load(
            (encoder_path,),
            KREA2_CLIP_TYPE,
            text_encoder_device,
        )
        loaded_vae = self._load_vae(vae, progress)
        return model, clip, loaded_vae

    def _require_krea2_model(self, model: object) -> None:
        """Reject non-Krea architectures before resolving large support files."""

        metadata = self._model_inspector.inspect(model)
        if metadata is not None and metadata.image_model == KREA2_IMAGE_MODEL:
            return
        raise ValueError(
            "Simple Load Krea 2 requires a diffusion model ComfyUI recognizes "
            "as Krea 2. Select a Krea 2 Raw or Turbo diffusion model."
        )

    def _resolve_text_encoder(
        self,
        selection: str,
        progress: ProgressReporter | None,
    ) -> Path:
        """Resolve auto and official selections or a local manual encoder."""

        artifact = (
            KREA2_QWEN3_VL_4B_FP8
            if selection == KREA2_AUTO_TEXT_ENCODER
            else KREA2_TEXT_ENCODER_ARTIFACTS.get(selection)
        )
        if artifact is not None:
            return self._resolver.resolve(artifact, progress).path
        path = self._folder_paths().get_full_path_or_raise(
            "text_encoders",
            selection,
        )
        return Path(str(path))

    def _load_vae(
        self,
        selection: str,
        progress: ProgressReporter | None,
    ) -> object:
        """Load the shared Qwen Image VAE automatically or a manual VAE."""

        if selection == AUTO_CHOICE:
            path = self._resolver.resolve(QWEN_IMAGE_VAE, progress).path
            return load_vae_path(path)
        return self._vae_loader.load_vae(selection)

    def _folder_paths(self) -> ModuleType:
        """Return the configured ComfyUI folder-path registry."""

        if self._folder_paths_module is not None:
            return self._folder_paths_module
        module: Any = importlib.import_module("folder_paths")
        if not isinstance(module, ModuleType):
            raise TypeError("folder_paths import did not return a module.")
        self._folder_paths_module = module
        return module


class DiffusionModelLoaderBoundary(Protocol):
    """Load one selected standalone diffusion model."""

    def load(self, diffusion_model: str, weight_dtype: str) -> object:
        """Return a loaded ComfyUI model patcher."""


class TextEncoderLoaderBoundary(Protocol):
    """Load one or more text-encoder files for a named Comfy CLIP type."""

    def load(
        self,
        paths: tuple[Path, ...],
        clip_type_name: str,
        device: str,
    ) -> object:
        """Return a loaded ComfyUI CLIP object."""


class DiffusionModelInspectorBoundary(Protocol):
    """Inspect architecture metadata from a loaded diffusion model."""

    def inspect(self, model: object) -> DiffusionModelMetadata | None:
        """Return normalized metadata when ComfyUI exposes it."""


class ClipTypeSupportBoundary(Protocol):
    """Validate that the host supports a required Comfy CLIP type."""

    def require(self, clip_type_name: str) -> None:
        """Raise when a required CLIP type is unavailable."""


class AutoModelResolverBoundary(Protocol):
    """Resolve a trusted catalog artifact to a verified local file."""

    def resolve(
        self,
        artifact: AutoModelArtifact,
        progress: ProgressReporter | None = None,
    ) -> AutoModelResolution:
        """Return a verified local artifact path."""


class VaeLoaderBoundary(Protocol):
    """Load one manually selected ComfyUI VAE."""

    def load_vae(self, vae_name: str) -> object:
        """Return a loaded ComfyUI VAE object."""
