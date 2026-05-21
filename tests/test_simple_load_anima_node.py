# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Simple Load Anima node declaration."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from simple_syrup.nodes.simple_load_anima import SimpleLoadAnima


class FakeFolderPaths(ModuleType):
    """Small folder_paths fake for node input declarations."""

    def __init__(self) -> None:
        """Create deterministic ComfyUI filename lists."""

        super().__init__("folder_paths")
        self.files = {
            "diffusion_models": ["anima.safetensors"],
            "text_encoders": ["qwen\\qwen_3_06b_base.safetensors"],
            "vae": ["qwen\\qwen_image_vae.safetensors"],
            "vae_approx": [],
        }

    def get_filename_list(self, folder_name: str) -> list[str]:
        """Return deterministic filenames for a model folder."""

        return self.files[folder_name]


def test_simple_load_anima_contract() -> None:
    """Simple Load Anima exposes MODEL, CLIP, and VAE sockets."""

    assert SimpleLoadAnima.RETURN_TYPES == ("MODEL", "CLIP", "VAE")
    assert SimpleLoadAnima.RETURN_NAMES == ("model", "clip", "vae")
    assert SimpleLoadAnima.FUNCTION == "load_models"
    assert SimpleLoadAnima.CATEGORY == "SimpleSyrup/Loaders"


def test_simple_load_anima_declares_expected_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input declarations mirror the combined ComfyUI loader controls."""

    monkeypatch.setitem(sys.modules, "folder_paths", FakeFolderPaths())

    input_types: dict[str, dict[str, tuple[Any, ...]]] = SimpleLoadAnima.INPUT_TYPES()
    required = input_types["required"]

    assert list(required) == [
        "diffusion_model",
        "diffusion_weight_dtype",
        "text_encoder",
        "text_encoder_device",
        "vae",
    ]
    assert required["diffusion_model"][0] == ["anima.safetensors"]
    assert required["diffusion_weight_dtype"][0] == [
        "default",
        "fp8_e4m3fn",
        "fp8_e4m3fn_fast",
        "fp8_e5m2",
    ]
    assert required["text_encoder"][0][0] == "auto"
    assert required["text_encoder"][1]["default"] == "auto"
    assert required["text_encoder_device"][0] == ["default", "cpu"]
    assert required["vae"][0][0] == "auto"
    assert required["vae"][1]["default"] == "auto"


def test_simple_load_anima_delegates_to_service() -> None:
    """Node execution delegates to the Anima loader service."""

    class FakeService:
        """Service double for node execution."""

        def __init__(self) -> None:
            """Create a recording service double."""

            self.kwargs: dict[str, object] | None = None

        def load_models(self, **kwargs: object) -> tuple[str, str, str]:
            """Record call arguments and return fixed outputs."""

            self.kwargs = kwargs
            return ("model", "clip", "vae")

    fake_service = FakeService()
    original = SimpleLoadAnima._service
    SimpleLoadAnima._service = fake_service  # type: ignore[assignment]
    try:
        result = SimpleLoadAnima().load_models(
            diffusion_model="anima.safetensors",
            diffusion_weight_dtype="default",
            text_encoder="auto",
            text_encoder_device="default",
            vae="auto",
        )
    finally:
        SimpleLoadAnima._service = original

    assert result == ("model", "clip", "vae")
    assert fake_service.kwargs is not None
    assert fake_service.kwargs["text_encoder"] == "auto"
    assert fake_service.kwargs["vae"] == "auto"
    assert "progress" in fake_service.kwargs
