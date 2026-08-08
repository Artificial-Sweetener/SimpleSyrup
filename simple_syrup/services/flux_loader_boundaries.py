# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define collaboration boundaries shared by the FLUX loading use cases."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..domain.flux_profiles import FluxModelProfile
from ..runtime.auto_model_artifact import AutoModelArtifact
from ..runtime.model_downloads import ProgressReporter


class DiffusionModelLoaderBoundary(Protocol):
    """Load a selected standalone diffusion model."""

    def load(self, diffusion_model: str, weight_dtype: str) -> object:
        """Load and return a ComfyUI model patcher."""


class TextEncoderLoaderBoundary(Protocol):
    """Load selected text encoders for a ComfyUI CLIP family."""

    def load(
        self,
        paths: tuple[Path, ...],
        clip_type_name: str,
        device: str,
    ) -> object:
        """Load and return a ComfyUI CLIP object."""


class FluxModelInspectorBoundary(Protocol):
    """Classify a loaded diffusion model from ComfyUI metadata."""

    def inspect(self, model: object) -> FluxModelProfile | None:
        """Return the detected FLUX profile when available."""


class FluxLoaderComponentsBoundary(Protocol):
    """Resolve and load selectable FLUX support components."""

    def resolve_text_encoder(
        self,
        selection: str,
        automatic_artifact: AutoModelArtifact | None,
        progress: ProgressReporter | None,
    ) -> Path:
        """Resolve an automatic or manual text encoder."""

    def load_vae(
        self,
        selection: str,
        automatic_artifact: AutoModelArtifact,
        progress: ProgressReporter | None,
    ) -> object:
        """Load an automatic or manual VAE."""
