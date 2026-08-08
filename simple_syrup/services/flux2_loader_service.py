# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Orchestrate FLUX.2 diffusion, profile-specific encoder, and VAE loading."""

from __future__ import annotations

from types import ModuleType

from ..domain.flux_profiles import FluxGeneration, FluxModelProfile
from ..runtime.auto_model_artifact import AutoModelArtifact
from ..runtime.diffusion_model_loader import DiffusionModelLoader
from ..runtime.flux_artifacts import FLUX2_TEXT_ENCODERS, FLUX2_VAE
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


class Flux2LoaderService:
    """Load FLUX.2 with its structurally selected encoder and common VAE."""

    def __init__(
        self,
        diffusion_loader: DiffusionModelLoaderBoundary | None = None,
        text_encoder_loader: TextEncoderLoaderBoundary | None = None,
        model_inspector: FluxModelInspectorBoundary | None = None,
        components: FluxLoaderComponentsBoundary | None = None,
        folder_paths_module: ModuleType | None = None,
    ) -> None:
        """Create a FLUX.2 loader with injectable architecture boundaries."""

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
        text_encoder: str,
        text_encoder_device: str,
        vae: str,
        progress: ProgressReporter | None = None,
    ) -> tuple[object, object, object]:
        """Return ComfyUI MODEL, CLIP, and VAE objects for FLUX.2."""

        model = self._diffusion_loader.load(
            diffusion_model,
            diffusion_weight_dtype,
        )
        profile = (
            self._require_flux2_profile(model)
            if AUTO_CHOICE in (text_encoder, vae)
            else None
        )
        encoder_artifact = self._select_encoder_artifact(text_encoder, profile)
        encoder_path = self._components.resolve_text_encoder(
            text_encoder,
            encoder_artifact,
            progress,
        )
        clip = self._text_encoder_loader.load(
            (encoder_path,),
            "FLUX2",
            text_encoder_device,
        )
        loaded_vae = self._components.load_vae(vae, FLUX2_VAE, progress)
        return model, clip, loaded_vae

    def _require_flux2_profile(self, model: object) -> FluxModelProfile:
        """Require structural FLUX.2 detection before automatic selection."""

        profile = self._model_inspector.inspect(model)
        if profile is not None and profile.generation is FluxGeneration.FLUX2:
            return profile
        raise ValueError(
            "Automatic FLUX.2 components require a diffusion model ComfyUI "
            "recognizes as FLUX.2. Select the text encoder and VAE manually "
            "to try an unrecognized or non-FLUX model."
        )

    def _select_encoder_artifact(
        self,
        selection: str,
        profile: FluxModelProfile | None,
    ) -> AutoModelArtifact | None:
        """Return the automatic profile artifact when auto is selected."""

        if selection != AUTO_CHOICE:
            return None
        if profile is None or profile.flux2_text_encoder is None:
            raise ValueError(
                "ComfyUI recognized FLUX.2 but its text-conditioning dimension "
                "does not match dev, Klein 4B, or Klein 9B/KV. Select the text "
                "encoder manually."
            )
        return FLUX2_TEXT_ENCODERS[profile.flux2_text_encoder]
