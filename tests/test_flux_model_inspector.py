# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for loaded ComfyUI FLUX model metadata inspection."""

from __future__ import annotations

from dataclasses import dataclass

from simple_syrup.domain.flux_profiles import (
    Flux2TextEncoderProfile,
    FluxGeneration,
)
from simple_syrup.runtime.flux_model_inspector import FluxModelInspector


@dataclass
class FakeModelConfig:
    """Minimal ComfyUI model-config double."""

    unet_config: object


class FakeModelPatcher:
    """Minimal ComfyUI model-patcher double."""

    def __init__(self, unet_config: object) -> None:
        """Store tensor-derived configuration independently of any filename."""

        self.model_config = FakeModelConfig(unet_config)

    def get_model_object(self, name: str) -> object:
        """Return the requested model configuration."""

        assert name == "model_config"
        return self.model_config


def test_inspector_detects_renamed_klein_finetune_from_loaded_config() -> None:
    """Architecture detection depends on loaded dimensions, not model filenames."""

    model = FakeModelPatcher({"image_model": "flux2", "context_in_dim": 12_288})

    profile = FluxModelInspector().inspect(model)

    assert profile is not None
    assert profile.generation is FluxGeneration.FLUX2
    assert profile.flux2_text_encoder is Flux2TextEncoderProfile.KLEIN_9B


def test_inspector_returns_none_without_model_patcher_metadata() -> None:
    """Manual loader paths can proceed when a host object cannot be inspected."""

    assert FluxModelInspector().inspect(object()) is None


def test_inspector_returns_none_for_malformed_unet_config() -> None:
    """Malformed dynamic host metadata is narrowed conservatively."""

    assert FluxModelInspector().inspect(FakeModelPatcher("not a mapping")) is None
