# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Orchestrate FLUX.1 diffusion, dual text-encoder, and VAE loading."""

from __future__ import annotations

from types import ModuleType

from ..domain.flux_profiles import FluxGeneration
from ..runtime.diffusion_model_loader import DiffusionModelLoader
from ..runtime.flux_artifacts import FLUX_CLIP_L, FLUX_T5_XXL, FLUX_VAE
from ..runtime.flux_model_inspector import FluxModelInspector
from ..runtime.model_downloads import ProgressReporter
from ..runtime.text_encoder_loader import TextEncoderLoader
from .flux_loader_boundaries import (
    DiffusionModelLoaderBoundary,
    FluxLoaderComponentsBoundary,
    FluxModelInspectorBoundary,
    TextEncoderLoaderBoundary,
)
from .flux_loader_components import AUTO_CHOICE, FluxLoaderComponents


class FluxLoaderService:
    """Load FLUX.1 with its required CLIP-L, T5-XXL, and VAE components."""

    def __init__(
        self,
        diffusion_loader: DiffusionModelLoaderBoundary | None = None,
        text_encoder_loader: TextEncoderLoaderBoundary | None = None,
        model_inspector: FluxModelInspectorBoundary | None = None,
        components: FluxLoaderComponentsBoundary | None = None,
        folder_paths_module: ModuleType | None = None,
    ) -> None:
        """Create a FLUX.1 loader with injectable architecture boundaries."""

        self._diffusion_loader = diffusion_loader or DiffusionModelLoader(
            folder_paths_module
        )
        self._text_encoder_loader = text_encoder_loader or TextEncoderLoader(
            folder_paths_module
        )
        self._model_inspector = model_inspector or FluxModelInspector()
        self._components = components or FluxLoaderComponents(
            folder_paths_module=folder_paths_module
        )

    def load_models(
        self,
        diffusion_model: str,
        diffusion_weight_dtype: str,
        clip_l: str,
        t5_xxl: str,
        text_encoder_device: str,
        vae: str,
        progress: ProgressReporter | None = None,
    ) -> tuple[object, object, object]:
        """Return ComfyUI MODEL, CLIP, and VAE objects for FLUX.1."""

        model = self._diffusion_loader.load(
            diffusion_model,
            diffusion_weight_dtype,
        )
        if AUTO_CHOICE in (clip_l, t5_xxl, vae):
            self._require_flux_profile(model)

        clip_l_path = self._components.resolve_text_encoder(
            clip_l,
            FLUX_CLIP_L,
            progress,
        )
        t5_xxl_path = self._components.resolve_text_encoder(
            t5_xxl,
            FLUX_T5_XXL,
            progress,
        )
        clip = self._text_encoder_loader.load(
            (clip_l_path, t5_xxl_path),
            "FLUX",
            text_encoder_device,
        )
        loaded_vae = self._components.load_vae(vae, FLUX_VAE, progress)
        return model, clip, loaded_vae

    def _require_flux_profile(self, model: object) -> None:
        """Require structural FLUX.1 detection before automatic selection."""

        profile = self._model_inspector.inspect(model)
        if profile is not None and profile.generation is FluxGeneration.FLUX:
            return
        raise ValueError(
            "Automatic FLUX.1 components require a diffusion model ComfyUI "
            "recognizes as FLUX.1. Select the text encoders and VAE manually "
            "to try an unrecognized or non-FLUX model."
        )
