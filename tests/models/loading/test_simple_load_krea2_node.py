# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Simple Load Krea 2 v3 node contract."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from simple_syrup.nodes_v3.simple_load_krea2 import SimpleLoadKrea2V3
from simple_syrup.runtime.krea2_artifacts import (
    KREA2_QWEN3_VL_4B_BF16,
    KREA2_QWEN3_VL_4B_FP8,
)
from simple_syrup.runtime.model_downloads import ComfyProgressReporter


class _FolderPaths(ModuleType):
    """Provide deterministic schema choices."""

    def __init__(self) -> None:
        """Create model lists including one official local encoder."""

        super().__init__("folder_paths")
        self._files = {
            "diffusion_models": ["krea2_turbo.safetensors"],
            "text_encoders": [
                "manual.safetensors",
                KREA2_QWEN3_VL_4B_BF16.filename,
            ],
            "vae": ["manual_vae.safetensors"],
            "vae_approx": [],
        }

    def get_filename_list(self, folder_name: str) -> list[str]:
        """Return configured files for one Comfy model folder."""

        return self._files[folder_name]


def test_krea2_schema_keeps_only_model_selection_non_advanced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The normal surface is one model choice with expert overrides collapsed."""

    monkeypatch.setitem(sys.modules, "folder_paths", _FolderPaths())

    schema = SimpleLoadKrea2V3.define_schema()
    inputs = {item.id: item for item in schema.inputs}

    assert schema.node_id == "SimpleSyrup.SimpleLoadKrea2"
    assert schema.display_name == "Simple Load Krea 2"
    assert [item.id for item in schema.inputs] == [
        "diffusion_model",
        "diffusion_weight_dtype",
        "text_encoder",
        "text_encoder_device",
        "vae",
    ]
    assert [output.io_type for output in schema.outputs] == ["MODEL", "CLIP", "VAE"]
    assert inputs["diffusion_model"].advanced is None
    assert all(item.advanced is True for item in schema.inputs[1:])
    assert inputs["text_encoder"].options[:3] == [
        "auto",
        KREA2_QWEN3_VL_4B_FP8.filename,
        KREA2_QWEN3_VL_4B_BF16.filename,
    ]
    assert inputs["text_encoder"].options.count(KREA2_QWEN3_VL_4B_BF16.filename) == 1
    assert inputs["text_encoder"].default == "auto"
    assert inputs["vae"].default == "auto"
    assert "checksum-pinned" in schema.description


def test_krea2_execute_supplies_visible_comfy_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Node execution forwards exact inputs and Comfy download progress."""

    class _Service:
        """Capture loader arguments and return fixed outputs."""

        def __init__(self) -> None:
            """Create an empty argument capture."""

            self.arguments: dict[str, object] = {}

        def load_models(self, **arguments: object) -> tuple[str, str, str]:
            """Record arguments and return deterministic outputs."""

            self.arguments = arguments
            return "model", "clip", "vae"

    service = _Service()
    monkeypatch.setattr(SimpleLoadKrea2V3, "_service", service)

    result = SimpleLoadKrea2V3.execute(
        "krea.safetensors",
        "default",
        KREA2_QWEN3_VL_4B_BF16.filename,
        "cpu",
        "auto",
    )

    assert result == ("model", "clip", "vae")
    assert service.arguments == {
        "diffusion_model": "krea.safetensors",
        "diffusion_weight_dtype": "default",
        "text_encoder": KREA2_QWEN3_VL_4B_BF16.filename,
        "text_encoder_device": "cpu",
        "vae": "auto",
        "progress": service.arguments["progress"],
    }
    assert isinstance(service.arguments["progress"], ComfyProgressReporter)
