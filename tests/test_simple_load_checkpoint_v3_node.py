# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Simple Load Checkpoint Comfy v3 wrapper."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from simple_syrup.nodes.simple_load_checkpoint import SimpleLoadCheckpoint
from simple_syrup.nodes_v3.simple_load_checkpoint import SimpleLoadCheckpointV3
from simple_syrup.runtime.checkpoint_loader import (
    CLIP_SKIP_DEFAULT,
    USE_CHECKPOINT_VAE_CHOICE,
)


class FakeFolderPaths(ModuleType):
    """Small folder_paths fake for v3 schema declarations."""

    def __init__(self) -> None:
        """Create deterministic ComfyUI filename lists."""

        super().__init__("folder_paths")
        self.files = {
            "checkpoints": ["model.safetensors"],
            "vae": ["manual_vae.safetensors"],
            "vae_approx": [],
        }

    def get_filename_list(self, folder_name: str) -> list[str]:
        """Return deterministic filenames for a model folder."""

        return self.files[folder_name]


def test_simple_load_checkpoint_v3_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The v3 schema mirrors the legacy checkpoint loader contract."""

    monkeypatch.setitem(sys.modules, "folder_paths", FakeFolderPaths())

    schema = SimpleLoadCheckpointV3.define_schema()

    assert schema.node_id == "SimpleSyrup.SimpleLoadCheckpoint"
    assert schema.display_name == "Simple Load Checkpoint"
    assert [input_item.id for input_item in schema.inputs] == [
        "ckpt_name",
        "vae_name",
        "clip_skip",
    ]
    assert schema.inputs[0].options == ["model.safetensors"]
    assert schema.inputs[1].options == [
        USE_CHECKPOINT_VAE_CHOICE,
        "manual_vae.safetensors",
        "pixel_space",
    ]
    assert schema.inputs[1].default == USE_CHECKPOINT_VAE_CHOICE
    assert schema.inputs[2].io_type == "BOOLEAN"
    assert schema.inputs[2].default is CLIP_SKIP_DEFAULT
    assert [output.id for output in schema.outputs] == ["model", "clip", "vae"]
    assert [output.io_type for output in schema.outputs] == ["MODEL", "CLIP", "VAE"]


def test_simple_load_checkpoint_v3_execute_forwards_to_legacy_loader(
    monkeypatch: Any,
) -> None:
    """The v3 wrapper forwards execution to the legacy loader."""

    class FakeService:
        """Service double for the legacy loader."""

        def __init__(self) -> None:
            """Initialize captured kwargs."""

            self.kwargs: dict[str, object] = {}

        def load_checkpoint(self, **kwargs: object) -> tuple[str, str, str]:
            """Return fixed checkpoint outputs."""

            self.kwargs = kwargs
            return ("model", "clip", "vae")

    fake_service = FakeService()
    monkeypatch.setattr(SimpleLoadCheckpoint, "_service", fake_service)

    assert SimpleLoadCheckpointV3.execute(
        "model.safetensors",
        USE_CHECKPOINT_VAE_CHOICE,
        True,
    ) == ("model", "clip", "vae")
    assert fake_service.kwargs == {
        "ckpt_name": "model.safetensors",
        "vae_name": USE_CHECKPOINT_VAE_CHOICE,
        "clip_skip": True,
    }
