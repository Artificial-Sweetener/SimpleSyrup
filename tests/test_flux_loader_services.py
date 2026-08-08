# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for separate FLUX.1 and FLUX.2 loading orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from simple_syrup.domain.flux_profiles import (
    Flux2TextEncoderProfile,
    FluxGeneration,
    FluxModelProfile,
)
from simple_syrup.runtime.auto_model_artifact import AutoModelArtifact
from simple_syrup.runtime.flux_artifacts import (
    FLUX2_DEV_TEXT_ENCODER,
    FLUX2_KLEIN_4B_TEXT_ENCODER,
    FLUX2_KLEIN_9B_TEXT_ENCODER,
    FLUX2_VAE,
    FLUX_CLIP_L,
    FLUX_T5_XXL,
    FLUX_VAE,
)
from simple_syrup.runtime.model_downloads import ProgressReporter
from simple_syrup.services.flux2_loader_service import Flux2LoaderService
from simple_syrup.services.flux_loader_components import AUTO_CHOICE
from simple_syrup.services.flux_loader_service import FluxLoaderService


class RecordingDiffusionLoader:
    """Record diffusion selection and return a fixed model object."""

    def __init__(self) -> None:
        """Create an empty call log."""

        self.calls: list[tuple[str, str]] = []
        self.model = object()

    def load(self, diffusion_model: str, weight_dtype: str) -> object:
        """Record the requested standalone model."""

        self.calls.append((diffusion_model, weight_dtype))
        return self.model


class RecordingInspector:
    """Return a configured model profile and record inspection calls."""

    def __init__(self, profile: FluxModelProfile | None) -> None:
        """Configure the profile returned by inspection."""

        self.profile = profile
        self.models: list[object] = []

    def inspect(self, model: object) -> FluxModelProfile | None:
        """Record and return the configured profile."""

        self.models.append(model)
        return self.profile


@dataclass
class RecordingComponents:
    """Record automatic/manual component orchestration requests."""

    encoder_requests: list[
        tuple[str, AutoModelArtifact | None, ProgressReporter | None]
    ] = field(default_factory=list)
    vae_requests: list[tuple[str, AutoModelArtifact, ProgressReporter | None]] = field(
        default_factory=list
    )

    def resolve_text_encoder(
        self,
        selection: str,
        automatic_artifact: AutoModelArtifact | None,
        progress: ProgressReporter | None,
    ) -> Path:
        """Record and return a deterministic encoder path."""

        self.encoder_requests.append((selection, automatic_artifact, progress))
        filename = (
            automatic_artifact.filename
            if selection == AUTO_CHOICE and automatic_artifact is not None
            else selection
        )
        return Path("C:/models/text_encoders") / filename

    def load_vae(
        self,
        selection: str,
        automatic_artifact: AutoModelArtifact,
        progress: ProgressReporter | None,
    ) -> object:
        """Record and return a deterministic VAE object."""

        self.vae_requests.append((selection, automatic_artifact, progress))
        return "vae"


@dataclass
class RecordingTextEncoderLoader:
    """Record text encoder paths, Comfy CLIP type, and device."""

    calls: list[tuple[tuple[Path, ...], str, str]] = field(default_factory=list)

    def load(
        self,
        paths: tuple[Path, ...],
        clip_type_name: str,
        device: str,
    ) -> object:
        """Record and return a fixed CLIP object."""

        self.calls.append((paths, clip_type_name, device))
        return "clip"


class IdentityProgress:
    """No-op progress reporter whose identity is asserted by tests."""

    def start(self, label: str, total: int | None) -> None:
        """Accept a progress start."""

    def advance(self, current: int, total: int | None) -> None:
        """Accept a progress update."""

    def finish(self) -> None:
        """Accept progress completion."""


def test_flux1_auto_loads_dual_encoders_and_generation_vae() -> None:
    """FLUX.1 auto resolves CLIP-L, T5-XXL, and ae through one progress path."""

    diffusion = RecordingDiffusionLoader()
    inspector = RecordingInspector(FluxModelProfile(FluxGeneration.FLUX))
    components = RecordingComponents()
    text_loader = RecordingTextEncoderLoader()
    progress = IdentityProgress()
    service = FluxLoaderService(diffusion, text_loader, inspector, components)

    result = service.load_models(
        "renamed-civit-finetune.safetensors",
        "fp8_e4m3fn_fast",
        AUTO_CHOICE,
        AUTO_CHOICE,
        "cpu",
        AUTO_CHOICE,
        progress,
    )

    assert result == (diffusion.model, "clip", "vae")
    assert components.encoder_requests == [
        (AUTO_CHOICE, FLUX_CLIP_L, progress),
        (AUTO_CHOICE, FLUX_T5_XXL, progress),
    ]
    assert components.vae_requests == [(AUTO_CHOICE, FLUX_VAE, progress)]
    assert text_loader.calls[0][1:] == ("FLUX", "cpu")


def test_flux1_auto_rejects_unrecognized_model_with_manual_guidance() -> None:
    """Unresolved FLUX.1 auto policy fails before any support artifact resolution."""

    components = RecordingComponents()
    service = FluxLoaderService(
        RecordingDiffusionLoader(),
        RecordingTextEncoderLoader(),
        RecordingInspector(None),
        components,
    )

    with pytest.raises(ValueError, match="Select the text encoders and VAE manually"):
        service.load_models(
            "unknown.safetensors",
            "default",
            AUTO_CHOICE,
            AUTO_CHOICE,
            "default",
            AUTO_CHOICE,
        )

    assert components.encoder_requests == []


def test_flux1_fully_manual_path_does_not_gatekeep_model_family() -> None:
    """Manual selections proceed without architecture detection."""

    inspector = RecordingInspector(None)
    components = RecordingComponents()
    text_loader = RecordingTextEncoderLoader()
    service = FluxLoaderService(
        RecordingDiffusionLoader(),
        text_loader,
        inspector,
        components,
    )

    result = service.load_models(
        "possibly-not-flux.safetensors",
        "default",
        "manual_clip.safetensors",
        "manual_t5.safetensors",
        "default",
        "manual_vae.safetensors",
    )

    assert result[1:] == ("clip", "vae")
    assert inspector.models == []
    assert text_loader.calls[0][1] == "FLUX"


@pytest.mark.parametrize(
    ("encoder_profile", "expected_artifact"),
    (
        (Flux2TextEncoderProfile.DEV, FLUX2_DEV_TEXT_ENCODER),
        (Flux2TextEncoderProfile.KLEIN_4B, FLUX2_KLEIN_4B_TEXT_ENCODER),
        (Flux2TextEncoderProfile.KLEIN_9B, FLUX2_KLEIN_9B_TEXT_ENCODER),
    ),
)
def test_flux2_auto_selects_encoder_for_loaded_architecture(
    encoder_profile: Flux2TextEncoderProfile,
    expected_artifact: AutoModelArtifact,
) -> None:
    """FLUX.2 auto supports dev, Klein 4B, and Klein 9B/KV profiles."""

    components = RecordingComponents()
    text_loader = RecordingTextEncoderLoader()
    progress = IdentityProgress()
    service = Flux2LoaderService(
        RecordingDiffusionLoader(),
        text_loader,
        RecordingInspector(FluxModelProfile(FluxGeneration.FLUX2, encoder_profile)),
        components,
    )

    result = service.load_models(
        "arbitrarily-renamed-finetune.safetensors",
        "default",
        AUTO_CHOICE,
        "default",
        AUTO_CHOICE,
        progress,
    )

    assert result[1:] == ("clip", "vae")
    assert components.encoder_requests == [(AUTO_CHOICE, expected_artifact, progress)]
    assert components.vae_requests == [(AUTO_CHOICE, FLUX2_VAE, progress)]
    assert text_loader.calls[0][1] == "FLUX2"


def test_flux2_unknown_width_requires_manual_text_encoder_only() -> None:
    """A recognized future FLUX.2 model can still auto-load its common VAE."""

    profile = FluxModelProfile(FluxGeneration.FLUX2)
    service = Flux2LoaderService(
        RecordingDiffusionLoader(),
        RecordingTextEncoderLoader(),
        RecordingInspector(profile),
        RecordingComponents(),
    )

    with pytest.raises(ValueError, match="Select the text encoder manually"):
        service.load_models(
            "future-flux2.safetensors",
            "default",
            AUTO_CHOICE,
            "default",
            AUTO_CHOICE,
        )

    components = RecordingComponents()
    manual_service = Flux2LoaderService(
        RecordingDiffusionLoader(),
        RecordingTextEncoderLoader(),
        RecordingInspector(profile),
        components,
    )
    manual_service.load_models(
        "future-flux2.safetensors",
        "default",
        "manual_encoder.safetensors",
        "default",
        AUTO_CHOICE,
    )

    assert components.encoder_requests[0][1] is None
    assert components.vae_requests[0][1] is FLUX2_VAE


def test_flux2_fully_manual_path_does_not_gatekeep_model_family() -> None:
    """Manual FLUX.2 selections proceed even when Comfy cannot classify the model."""

    inspector = RecordingInspector(None)
    text_loader = RecordingTextEncoderLoader()
    service = Flux2LoaderService(
        RecordingDiffusionLoader(),
        text_loader,
        inspector,
        RecordingComponents(),
    )

    service.load_models(
        "non-flux-or-unknown.safetensors",
        "default",
        "manual_encoder.safetensors",
        "default",
        "manual_vae.safetensors",
    )

    assert inspector.models == []
    assert text_loader.calls[0][1] == "FLUX2"


def test_flux2_auto_rejects_a_detected_flux1_model() -> None:
    """Cross-generation automatic selection fails with a manual escape hatch."""

    service = Flux2LoaderService(
        RecordingDiffusionLoader(),
        RecordingTextEncoderLoader(),
        RecordingInspector(FluxModelProfile(FluxGeneration.FLUX)),
        RecordingComponents(),
    )

    with pytest.raises(ValueError, match="recognizes as FLUX.2"):
        service.load_models(
            "flux1.safetensors",
            "default",
            AUTO_CHOICE,
            "default",
            AUTO_CHOICE,
        )
