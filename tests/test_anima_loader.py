# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Anima combined model loader service."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType

import pytest
import torch

from simple_syrup.runtime.anima_loader import (
    AUTO_CHOICE,
    AnimaLoaderService,
)
from simple_syrup.runtime.auto_model_resolver import AutoModelResolution
from simple_syrup.runtime.model_catalog import AutoModelArtifact
from simple_syrup.runtime.vae_loader import vae_choices


@dataclass
class FakeComfyState:
    """Recorded calls into fake ComfyUI loader APIs."""

    diffusion_calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    clip_calls: list[dict[str, object]] = field(default_factory=list)
    vae_paths: list[str] = field(default_factory=list)


class FakeFolderPaths(ModuleType):
    """Folder paths fake with ComfyUI loader methods."""

    def __init__(self, models_dir: Path) -> None:
        """Create fake folder state."""

        super().__init__("folder_paths")
        self.models_dir = str(models_dir)
        self.files: dict[str, list[str]] = {
            "diffusion_models": ["anima.safetensors"],
            "text_encoders": ["manual_clip.safetensors"],
            "vae": ["manual_vae.safetensors"],
            "vae_approx": [],
            "embeddings": [],
        }

    def get_filename_list(self, folder_name: str) -> list[str]:
        """Return configured relative filenames."""

        return self.files.get(folder_name, [])

    def get_folder_paths(self, folder_name: str) -> list[str]:
        """Return the conventional folder path for a model type."""

        return [str(Path(self.models_dir) / folder_name)]

    def get_full_path_or_raise(self, folder_name: str, filename: str) -> str:
        """Return a deterministic absolute path for a relative filename."""

        return str(Path(self.models_dir) / folder_name / filename)


class FakeResolver:
    """Resolver fake returning paths for Anima auto artifacts."""

    def __init__(self, text_encoder_path: Path, vae_path: Path) -> None:
        """Create a resolver with deterministic paths."""

        self.text_encoder_path = text_encoder_path
        self.vae_path = vae_path
        self.requests: list[str] = []

    def resolve(
        self,
        artifact: AutoModelArtifact,
        progress: object | None = None,
    ) -> AutoModelResolution:
        """Record and resolve one artifact."""

        del progress
        self.requests.append(artifact.cache_id)
        if artifact.folder_name == "text_encoders":
            return AutoModelResolution(self.text_encoder_path, "cached")
        return AutoModelResolution(self.vae_path, "cached")


def test_loader_maps_diffusion_weight_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diffusion dtype options match ComfyUI's UNETLoader behavior."""

    comfy_state = _install_fake_comfy(monkeypatch)
    folder_paths = FakeFolderPaths(tmp_path / "models")
    service = AnimaLoaderService(
        resolver=FakeResolver(
            tmp_path / "clip.safetensors", tmp_path / "vae.safetensors"
        ),
        folder_paths_module=folder_paths,
    )

    service.load_models(
        "anima.safetensors",
        "fp8_e4m3fn_fast",
        "manual_clip.safetensors",
        "default",
        "manual_vae.safetensors",
    )

    assert comfy_state.diffusion_calls[0][1] == {
        "dtype": torch.float8_e4m3fn,
        "fp8_optimizations": True,
    }


def test_loader_maps_clip_cpu_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLIP cpu device option matches ComfyUI's CLIPLoader behavior."""

    comfy_state = _install_fake_comfy(monkeypatch)
    folder_paths = FakeFolderPaths(tmp_path / "models")
    service = AnimaLoaderService(
        resolver=FakeResolver(
            tmp_path / "clip.safetensors", tmp_path / "vae.safetensors"
        ),
        folder_paths_module=folder_paths,
    )

    service.load_models(
        "anima.safetensors",
        "default",
        "manual_clip.safetensors",
        "cpu",
        "manual_vae.safetensors",
    )

    model_options = comfy_state.clip_calls[0]["model_options"]
    assert model_options == {
        "load_device": torch.device("cpu"),
        "offload_device": torch.device("cpu"),
    }
    clip_type = comfy_state.clip_calls[0]["clip_type"]
    assert isinstance(clip_type, Enum)
    assert clip_type.name == "STABLE_DIFFUSION"


def test_loader_uses_auto_resolver_for_auto_choices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto text encoder and VAE selections use resolved auto paths."""

    comfy_state = _install_fake_comfy(monkeypatch)
    folder_paths = FakeFolderPaths(tmp_path / "models")
    resolver = FakeResolver(
        tmp_path / "models" / "text_encoders" / "qwen" / "qwen_3_06b_base.safetensors",
        tmp_path / "models" / "vae" / "qwen" / "qwen_image_vae.safetensors",
    )
    service = AnimaLoaderService(resolver=resolver, folder_paths_module=folder_paths)

    service.load_models(
        "anima.safetensors",
        "default",
        AUTO_CHOICE,
        "default",
        AUTO_CHOICE,
    )

    assert resolver.requests == ["anima_qwen_text_encoder", "anima_qwen_vae"]
    assert comfy_state.clip_calls[0]["ckpt_paths"] == [str(resolver.text_encoder_path)]
    assert comfy_state.vae_paths == [str(resolver.vae_path)]


def test_loader_returns_model_clip_and_vae(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Service returns the combined ComfyUI loader outputs."""

    comfy_state = _install_fake_comfy(monkeypatch)
    folder_paths = FakeFolderPaths(tmp_path / "models")
    service = AnimaLoaderService(
        resolver=FakeResolver(
            tmp_path / "clip.safetensors", tmp_path / "vae.safetensors"
        ),
        folder_paths_module=folder_paths,
    )

    result = service.load_models(
        "anima.safetensors",
        "default",
        "manual_clip.safetensors",
        "default",
        "manual_vae.safetensors",
    )

    assert result[0] == "model"
    assert result[1] == "clip"
    assert result[2] is not None
    assert comfy_state.vae_paths == [
        str(tmp_path / "models" / "vae" / "manual_vae.safetensors")
    ]


def test_vae_choices_match_comfy_special_choices(tmp_path: Path) -> None:
    """VAE choices include ComfyUI TAESD and pixel-space entries."""

    folder_paths = FakeFolderPaths(tmp_path / "models")
    folder_paths.files["vae"] = ["manual_vae.safetensors"]
    folder_paths.files["vae_approx"] = [
        "taesd_encoder.pth",
        "taesd_decoder.pth",
        "taehv.pth",
    ]

    assert vae_choices(folder_paths) == [
        "manual_vae.safetensors",
        "taehv.pth",
        "taesd",
        "pixel_space",
    ]


def _install_fake_comfy(monkeypatch: pytest.MonkeyPatch) -> FakeComfyState:
    """Install fake ComfyUI modules and return the fake comfy.sd module."""

    class FakeCLIPType(Enum):
        """Small CLIPType enum fake."""

        STABLE_DIFFUSION = 1
        QWEN_IMAGE = 2

    class FakeVAE:
        """Small VAE fake."""

        def __init__(
            self, sd: dict[str, object], metadata: object | None = None
        ) -> None:
            """Record constructor inputs."""

            del sd, metadata

        def throw_exception_if_invalid(self) -> None:
            """Accept validation."""

    comfy_module = ModuleType("comfy")
    comfy_sd = ModuleType("comfy.sd")
    comfy_utils = ModuleType("comfy.utils")
    state = FakeComfyState()
    comfy_sd.CLIPType = FakeCLIPType  # type: ignore[attr-defined]

    def load_diffusion_model(
        path: str,
        model_options: dict[str, object],
    ) -> str:
        """Record diffusion model calls."""

        state.diffusion_calls.append((path, model_options))
        return "model"

    def load_clip(
        ckpt_paths: list[str],
        embedding_directory: list[str],
        clip_type: FakeCLIPType,
        model_options: dict[str, object],
    ) -> str:
        """Record CLIP loader calls."""

        state.clip_calls.append(
            {
                "ckpt_paths": ckpt_paths,
                "embedding_directory": embedding_directory,
                "clip_type": clip_type,
                "model_options": model_options,
            }
        )
        return "clip"

    def load_torch_file(
        path: str,
        return_metadata: bool = False,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Record VAE file loads."""

        del return_metadata
        state.vae_paths.append(path)
        return {}, {}

    comfy_sd.load_diffusion_model = load_diffusion_model  # type: ignore[attr-defined]
    comfy_sd.load_clip = load_clip  # type: ignore[attr-defined]
    comfy_sd.VAE = FakeVAE  # type: ignore[attr-defined]
    comfy_utils.load_torch_file = load_torch_file  # type: ignore[attr-defined]
    comfy_module.sd = comfy_sd  # type: ignore[attr-defined]
    comfy_module.utils = comfy_utils  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils)
    return state
