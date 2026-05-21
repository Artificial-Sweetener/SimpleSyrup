# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for checkpoint loading with optional VAE replacement."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

from simple_syrup.runtime.checkpoint_loader import (
    CLIP_SKIP_DEFAULT,
    CLIP_SKIP_LAYER,
    USE_CHECKPOINT_VAE_CHOICE,
    CheckpointLoaderService,
)


class FakeClip:
    """CLIP double that records clone and layer selection behavior."""

    def __init__(self, name: str = "checkpoint_clip") -> None:
        """Create a CLIP double with no selected layer."""

        self.name = name
        self.layer: int | None = None
        self.clone_count = 0

    def clone(self) -> FakeClip:
        """Return an independent CLIP double and record the clone call."""

        self.clone_count += 1
        return FakeClip(f"{self.name}_clone")

    def clip_layer(self, layer: int) -> None:
        """Record the selected CLIP layer."""

        self.layer = layer


@dataclass
class FakeComfyState:
    """Recorded calls into fake ComfyUI checkpoint APIs."""

    checkpoint_clip: FakeClip = field(default_factory=FakeClip)
    checkpoint_calls: list[dict[str, object]] = field(default_factory=list)


class FakeFolderPaths(ModuleType):
    """Folder paths fake with checkpoint loader methods."""

    def __init__(self, models_dir: Path) -> None:
        """Create fake checkpoint folder state."""

        super().__init__("folder_paths")
        self.models_dir = str(models_dir)

    def get_folder_paths(self, folder_name: str) -> list[str]:
        """Return a deterministic folder path for a model type."""

        return [str(Path(self.models_dir) / folder_name)]

    def get_full_path_or_raise(self, folder_name: str, filename: str) -> str:
        """Return a deterministic absolute path for a relative filename."""

        return str(Path(self.models_dir) / folder_name / filename)


class FakeVaeLoader:
    """External VAE loader double."""

    def __init__(self) -> None:
        """Create call recording state."""

        self.requests: list[str] = []

    def load_vae(self, vae_name: str) -> object:
        """Return a fixed external VAE object."""

        self.requests.append(vae_name)
        if vae_name == "missing_vae.safetensors":
            raise ValueError("missing VAE")
        return "external_vae"


def test_checkpoint_loader_returns_checkpoint_vae_with_clip_skip_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The false selection preserves the loaded CLIP and checkpoint VAE."""

    state = _install_fake_comfy(monkeypatch)
    vae_loader = FakeVaeLoader()
    folder_paths = FakeFolderPaths(tmp_path / "models")
    service = CheckpointLoaderService(
        folder_paths_module=folder_paths,
        vae_loader=vae_loader,
    )

    result = service.load_checkpoint(
        ckpt_name="model.safetensors",
        vae_name=USE_CHECKPOINT_VAE_CHOICE,
        clip_skip=CLIP_SKIP_DEFAULT,
    )

    assert result == ("checkpoint_model", state.checkpoint_clip, "checkpoint_vae")
    assert state.checkpoint_clip.clone_count == 0
    assert state.checkpoint_clip.layer is None
    assert vae_loader.requests == []
    assert state.checkpoint_calls == [
        {
            "ckpt_path": str(tmp_path / "models" / "checkpoints" / "model.safetensors"),
            "output_vae": True,
            "output_clip": True,
            "embedding_directory": [str(tmp_path / "models" / "embeddings")],
        }
    ]


def test_checkpoint_loader_applies_clip_skip_to_checkpoint_vae_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The true selection applies Comfy's -2 layer to a cloned CLIP."""

    state = _install_fake_comfy(monkeypatch)
    service = CheckpointLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path / "models"),
        vae_loader=FakeVaeLoader(),
    )

    result = service.load_checkpoint(
        ckpt_name="model.safetensors",
        vae_name=USE_CHECKPOINT_VAE_CHOICE,
        clip_skip=True,
    )

    assert result[0] == "checkpoint_model"
    assert result[1] is not state.checkpoint_clip
    assert isinstance(result[1], FakeClip)
    assert result[1].name == "checkpoint_clip_clone"
    assert result[1].layer == CLIP_SKIP_LAYER
    assert result[2] == "checkpoint_vae"
    assert state.checkpoint_clip.clone_count == 1
    assert state.checkpoint_clip.layer is None


def test_checkpoint_loader_replaces_only_vae(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External VAE selection replaces only the VAE output."""

    state = _install_fake_comfy(monkeypatch)
    vae_loader = FakeVaeLoader()
    service = CheckpointLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path / "models"),
        vae_loader=vae_loader,
    )

    result = service.load_checkpoint(
        ckpt_name="model.safetensors",
        vae_name="external_vae.safetensors",
        clip_skip=False,
    )

    assert result == ("checkpoint_model", state.checkpoint_clip, "external_vae")
    assert state.checkpoint_clip.clone_count == 0
    assert vae_loader.requests == ["external_vae.safetensors"]


def test_checkpoint_loader_applies_clip_skip_to_external_vae_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External VAE replacement remains independent of clip skip."""

    state = _install_fake_comfy(monkeypatch)
    vae_loader = FakeVaeLoader()
    service = CheckpointLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path / "models"),
        vae_loader=vae_loader,
    )

    result = service.load_checkpoint(
        ckpt_name="model.safetensors",
        vae_name="external_vae.safetensors",
        clip_skip=True,
    )

    assert result[0] == "checkpoint_model"
    assert result[1] is not state.checkpoint_clip
    assert isinstance(result[1], FakeClip)
    assert result[1].layer == CLIP_SKIP_LAYER
    assert result[2] == "external_vae"
    assert vae_loader.requests == ["external_vae.safetensors"]


def test_checkpoint_loader_does_not_swallow_external_vae_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External VAE load failures surface to ComfyUI callers."""

    _install_fake_comfy(monkeypatch)
    service = CheckpointLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path / "models"),
        vae_loader=FakeVaeLoader(),
    )

    with pytest.raises(ValueError, match="missing VAE"):
        service.load_checkpoint(
            ckpt_name="model.safetensors",
            vae_name="missing_vae.safetensors",
            clip_skip=False,
        )


def test_checkpoint_loader_rejects_non_boolean_clip_skip(tmp_path: Path) -> None:
    """Non-boolean clip skip values fail before checkpoint loading."""

    service = CheckpointLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path / "models"),
        vae_loader=FakeVaeLoader(),
    )

    with pytest.raises(TypeError, match="clip_skip.*boolean"):
        service.load_checkpoint(
            ckpt_name="model.safetensors",
            vae_name=USE_CHECKPOINT_VAE_CHOICE,
            clip_skip="on",  # type: ignore[arg-type]
        )


def _install_fake_comfy(monkeypatch: pytest.MonkeyPatch) -> FakeComfyState:
    """Install fake ComfyUI checkpoint modules."""

    comfy_module = ModuleType("comfy")
    comfy_sd = ModuleType("comfy.sd")
    state = FakeComfyState()

    def load_checkpoint_guess_config(
        ckpt_path: str,
        output_vae: bool,
        output_clip: bool,
        embedding_directory: list[str],
    ) -> tuple[object, object, object, object]:
        """Record checkpoint load calls and return fixed outputs."""

        state.checkpoint_calls.append(
            {
                "ckpt_path": ckpt_path,
                "output_vae": output_vae,
                "output_clip": output_clip,
                "embedding_directory": embedding_directory,
            }
        )
        return (
            "checkpoint_model",
            state.checkpoint_clip,
            "checkpoint_vae",
            "ignored_clipvision",
        )

    comfy_sd.load_checkpoint_guess_config = load_checkpoint_guess_config  # type: ignore[attr-defined]
    comfy_module.sd = comfy_sd  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)
    return state
