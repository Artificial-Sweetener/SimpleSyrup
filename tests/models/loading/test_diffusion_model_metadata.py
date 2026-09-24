# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for shared loaded diffusion-model metadata inspection."""

from __future__ import annotations

from dataclasses import dataclass

from simple_syrup.runtime.diffusion_model_metadata import (
    DiffusionModelMetadata,
    DiffusionModelMetadataInspector,
)


@dataclass
class _ModelConfig:
    """Expose one fake tensor-derived UNet configuration."""

    unet_config: object


class _ModelPatcher:
    """Expose a model config through ComfyUI's model-patcher surface."""

    def __init__(self, unet_config: object) -> None:
        """Store the fake UNet configuration."""

        self._config = _ModelConfig(unet_config)

    def get_model_object(self, name: str) -> object:
        """Return only the requested model configuration."""

        assert name == "model_config"
        return self._config


def test_metadata_inspector_normalizes_krea_architecture() -> None:
    """Krea detection uses loaded tensor metadata rather than its filename."""

    result = DiffusionModelMetadataInspector().inspect(
        _ModelPatcher({"image_model": "krea2", "context_in_dim": 30_720})
    )

    assert result == DiffusionModelMetadata("krea2", 30_720)


def test_metadata_inspector_rejects_unavailable_and_malformed_surfaces() -> None:
    """Dynamic host values fail closed when structural metadata is unavailable."""

    inspector = DiffusionModelMetadataInspector()

    assert inspector.inspect(object()) is None
    assert inspector.inspect(_ModelPatcher("not a mapping")) is None
