# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve automatic or manual component selections for FLUX loaders."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from ..runtime.auto_model_artifact import AutoModelArtifact
from ..runtime.auto_model_resolver import AutoModelResolution, AutoModelResolver
from ..runtime.model_downloads import ProgressReporter
from ..runtime.vae_loader import VaeLoaderService, load_vae_path

AUTO_CHOICE = "auto"


class AutoModelResolverBoundary(Protocol):
    """Resolve a trusted automatic artifact to a local path."""

    def resolve(
        self,
        artifact: AutoModelArtifact,
        progress: ProgressReporter | None = None,
    ) -> AutoModelResolution:
        """Resolve one automatic artifact."""


class VaeLoaderBoundary(Protocol):
    """Load a manually selected ComfyUI VAE choice."""

    def load_vae(self, vae_name: str) -> object:
        """Load one manual VAE selection."""


class FluxLoaderComponents:
    """Own component selection, automatic resolution, and VAE loading."""

    def __init__(
        self,
        resolver: AutoModelResolverBoundary | None = None,
        vae_loader: VaeLoaderBoundary | None = None,
        folder_paths_module: ModuleType | None = None,
    ) -> None:
        """Create component orchestration with injectable external boundaries."""

        self._folder_paths_module = folder_paths_module
        self._resolver = resolver or AutoModelResolver(
            folder_paths_module=folder_paths_module
        )
        self._vae_loader = vae_loader or VaeLoaderService(folder_paths_module)

    def resolve_text_encoder(
        self,
        selection: str,
        automatic_artifact: AutoModelArtifact | None,
        progress: ProgressReporter | None,
    ) -> Path:
        """Resolve an automatic artifact or a manual text-encoder selection."""

        if selection == AUTO_CHOICE:
            if automatic_artifact is None:
                raise ValueError("Automatic text encoder artifact was not resolved.")
            return self._resolver.resolve(automatic_artifact, progress).path
        path = self._folder_paths().get_full_path_or_raise(
            "text_encoders",
            selection,
        )
        return Path(str(path))

    def load_vae(
        self,
        selection: str,
        automatic_artifact: AutoModelArtifact,
        progress: ProgressReporter | None,
    ) -> object:
        """Resolve and load an automatic VAE or load a manual VAE selection."""

        if selection == AUTO_CHOICE:
            path = self._resolver.resolve(automatic_artifact, progress).path
            return load_vae_path(path)
        return self._vae_loader.load_vae(selection)

    def _folder_paths(self) -> ModuleType:
        """Return the configured ComfyUI folder paths module."""

        if self._folder_paths_module is not None:
            return self._folder_paths_module
        module: Any = importlib.import_module("folder_paths")
        if not isinstance(module, ModuleType):
            raise TypeError("folder_paths import did not return a module.")
        self._folder_paths_module = module
        return module
