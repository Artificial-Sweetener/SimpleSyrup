# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Simple Load Checkpoint node declaration."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from simple_syrup.nodes.simple_load_checkpoint import SimpleLoadCheckpoint
from simple_syrup.runtime.checkpoint_loader import (
    CLIP_SKIP_DEFAULT,
    USE_CHECKPOINT_VAE_CHOICE,
)


class FakeFolderPaths(ModuleType):
    """Small folder_paths fake for checkpoint node input declarations."""

    def __init__(self) -> None:
        """Create deterministic ComfyUI filename lists."""

        super().__init__("folder_paths")
        self.files = {
            "checkpoints": ["model.safetensors"],
            "vae": ["manual_vae.safetensors", USE_CHECKPOINT_VAE_CHOICE],
            "vae_approx": [],
        }

    def get_filename_list(self, folder_name: str) -> list[str]:
        """Return deterministic filenames for a model folder."""

        return self.files[folder_name]


def test_simple_load_checkpoint_contract() -> None:
    """Simple Load Checkpoint exposes MODEL, CLIP, and VAE sockets."""

    assert SimpleLoadCheckpoint.RETURN_TYPES == ("MODEL", "CLIP", "VAE")
    assert SimpleLoadCheckpoint.RETURN_NAMES == ("model", "clip", "vae")
    assert SimpleLoadCheckpoint.FUNCTION == "load_checkpoint"
    assert SimpleLoadCheckpoint.CATEGORY == "SimpleSyrup/Loaders"


def test_simple_load_checkpoint_declares_expected_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input declarations mirror checkpoint loading plus VAE override."""

    monkeypatch.setitem(sys.modules, "folder_paths", FakeFolderPaths())

    input_types: dict[str, dict[str, tuple[Any, ...]]] = (
        SimpleLoadCheckpoint.INPUT_TYPES()
    )
    required = input_types["required"]

    assert list(required) == ["ckpt_name", "vae_name", "clip_skip"]
    assert required["ckpt_name"][0] == ["model.safetensors"]
    assert required["vae_name"][0] == [
        USE_CHECKPOINT_VAE_CHOICE,
        "manual_vae.safetensors",
        "pixel_space",
    ]
    assert required["vae_name"][1]["default"] == USE_CHECKPOINT_VAE_CHOICE
    assert required["clip_skip"][0] == "BOOLEAN"
    assert required["clip_skip"][1]["default"] is CLIP_SKIP_DEFAULT


def test_simple_load_checkpoint_delegates_to_service() -> None:
    """Node execution delegates to the checkpoint loader service."""

    class FakeService:
        """Service double for node execution."""

        def __init__(self) -> None:
            """Create a recording service double."""

            self.kwargs: dict[str, object] | None = None

        def load_checkpoint(self, **kwargs: object) -> tuple[str, str, str]:
            """Record call arguments and return fixed outputs."""

            self.kwargs = kwargs
            return ("model", "clip", "vae")

    fake_service = FakeService()
    original = SimpleLoadCheckpoint._service
    SimpleLoadCheckpoint._service = fake_service  # type: ignore[assignment]
    try:
        result = SimpleLoadCheckpoint().load_checkpoint(
            ckpt_name="model.safetensors",
            vae_name=USE_CHECKPOINT_VAE_CHOICE,
            clip_skip=True,
        )
    finally:
        SimpleLoadCheckpoint._service = original

    assert result == ("model", "clip", "vae")
    assert fake_service.kwargs == {
        "ckpt_name": "model.safetensors",
        "vae_name": USE_CHECKPOINT_VAE_CHOICE,
        "clip_skip": True,
    }
