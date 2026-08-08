# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ComfyUI runtime adapters used by the FLUX loader services."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType

import pytest
import torch

import simple_syrup.services.flux_loader_components as components_module
from simple_syrup.runtime.auto_model_artifact import AutoModelArtifact
from simple_syrup.runtime.auto_model_resolver import AutoModelResolution
from simple_syrup.runtime.diffusion_model_loader import DiffusionModelLoader
from simple_syrup.runtime.model_downloads import ProgressReporter
from simple_syrup.runtime.text_encoder_loader import TextEncoderLoader
from simple_syrup.services.flux_loader_components import (
    AUTO_CHOICE,
    FluxLoaderComponents,
)


class FakeFolderPaths(ModuleType):
    """Resolve manual models to deterministic absolute test paths."""

    def __init__(self, root: Path) -> None:
        """Create folder paths rooted in the test directory."""

        super().__init__("folder_paths")
        self.root = root

    def get_full_path_or_raise(self, folder_name: str, filename: str) -> str:
        """Return an absolute path for a named ComfyUI model selection."""

        return str(self.root / folder_name / filename)

    def get_folder_paths(self, folder_name: str) -> list[str]:
        """Return registered paths for embeddings and model folders."""

        return [str(self.root / folder_name)]


@dataclass
class FakeComfyState:
    """Record calls made through the ComfyUI loading adapters."""

    diffusion_calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    clip_calls: list[dict[str, object]] = field(default_factory=list)


class FakeCLIPType(Enum):
    """Represent the ComfyUI CLIP families used by FLUX loaders."""

    FLUX = 1
    FLUX2 = 2


class IdentityProgress:
    """No-op progress reporter used to verify identity propagation."""

    def start(self, label: str, total: int | None) -> None:
        """Accept a progress start."""

    def advance(self, current: int, total: int | None) -> None:
        """Accept a progress update."""

    def finish(self) -> None:
        """Accept progress completion."""


class RecordingResolver:
    """Resolve trusted artifacts to deterministic local files."""

    def __init__(self, root: Path) -> None:
        """Create an empty automatic-resolution log."""

        self.root = root
        self.calls: list[tuple[AutoModelArtifact, ProgressReporter | None]] = []

    def resolve(
        self,
        artifact: AutoModelArtifact,
        progress: ProgressReporter | None = None,
    ) -> AutoModelResolution:
        """Record and return one automatic artifact path."""

        self.calls.append((artifact, progress))
        return AutoModelResolution(self.root / artifact.filename, "cached")


class RecordingVaeLoader:
    """Record manual VAE choices."""

    def __init__(self) -> None:
        """Create an empty manual VAE log."""

        self.calls: list[str] = []

    def load_vae(self, vae_name: str) -> object:
        """Record and return a manual VAE object."""

        self.calls.append(vae_name)
        return "manual vae"


def test_diffusion_adapter_matches_comfy_unet_dtype_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone diffusion loading forwards ComfyUI's fast FP8 options."""

    state = _install_fake_comfy(monkeypatch)
    loader = DiffusionModelLoader(FakeFolderPaths(tmp_path))

    model = loader.load("flux.safetensors", "fp8_e4m3fn_fast")

    assert model == "model"
    assert state.diffusion_calls == [
        (
            str(tmp_path / "diffusion_models" / "flux.safetensors"),
            {"dtype": torch.float8_e4m3fn, "fp8_optimizations": True},
        )
    ]


def test_diffusion_adapter_rejects_unknown_dtype(tmp_path: Path) -> None:
    """Invalid workflow-facing dtype values fail before ComfyUI loading."""

    with pytest.raises(ValueError, match="diffusion_weight_dtype"):
        DiffusionModelLoader(FakeFolderPaths(tmp_path)).load(
            "flux.safetensors",
            "invalid",
        )


def test_text_encoder_adapter_loads_dual_flux_encoders_on_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dual FLUX encoders use CLIPType.FLUX and explicit CPU placement."""

    state = _install_fake_comfy(monkeypatch)
    paths = (tmp_path / "clip.safetensors", tmp_path / "t5.safetensors")

    clip = TextEncoderLoader(FakeFolderPaths(tmp_path)).load(paths, "FLUX", "cpu")

    assert clip == "clip"
    assert state.clip_calls[0]["paths"] == [str(path) for path in paths]
    assert state.clip_calls[0]["clip_type"] is FakeCLIPType.FLUX
    assert state.clip_calls[0]["model_options"] == {
        "load_device": torch.device("cpu"),
        "offload_device": torch.device("cpu"),
    }


def test_text_encoder_adapter_rejects_unknown_device(tmp_path: Path) -> None:
    """Invalid encoder device values fail before ComfyUI loading."""

    with pytest.raises(ValueError, match="text_encoder_device"):
        TextEncoderLoader(FakeFolderPaths(tmp_path)).load(
            (tmp_path / "encoder.safetensors",),
            "FLUX2",
            "cuda:7",
        )


def test_component_orchestrator_preserves_auto_progress_and_manual_escape_hatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto resolution receives progress while manual selections remain direct."""

    artifact = _artifact()
    resolver = RecordingResolver(tmp_path / "resolved")
    vae_loader = RecordingVaeLoader()
    progress = IdentityProgress()
    loaded_auto_paths: list[Path] = []

    def load_automatic_vae(path: Path) -> object:
        """Record and return an automatically resolved VAE path."""

        loaded_auto_paths.append(path)
        return "auto vae"

    monkeypatch.setattr(
        components_module,
        "load_vae_path",
        load_automatic_vae,
    )
    components = FluxLoaderComponents(
        resolver,
        vae_loader,
        FakeFolderPaths(tmp_path),
    )

    automatic_path = components.resolve_text_encoder(
        AUTO_CHOICE,
        artifact,
        progress,
    )
    manual_path = components.resolve_text_encoder(
        "manual.safetensors",
        None,
        progress,
    )
    automatic_vae = components.load_vae(AUTO_CHOICE, artifact, progress)
    manual_vae = components.load_vae("manual_vae.safetensors", artifact, progress)

    assert automatic_path == tmp_path / "resolved" / artifact.filename
    assert manual_path == tmp_path / "text_encoders" / "manual.safetensors"
    assert resolver.calls == [(artifact, progress), (artifact, progress)]
    assert loaded_auto_paths == [tmp_path / "resolved" / artifact.filename]
    assert automatic_vae == "auto vae"
    assert manual_vae == "manual vae"
    assert vae_loader.calls == ["manual_vae.safetensors"]


def test_component_orchestrator_rejects_unresolved_auto_encoder(tmp_path: Path) -> None:
    """An automatic encoder request cannot silently fall through to a manual path."""

    components = FluxLoaderComponents(
        RecordingResolver(tmp_path),
        RecordingVaeLoader(),
        FakeFolderPaths(tmp_path),
    )

    with pytest.raises(ValueError, match="artifact was not resolved"):
        components.resolve_text_encoder(AUTO_CHOICE, None, None)


def _install_fake_comfy(monkeypatch: pytest.MonkeyPatch) -> FakeComfyState:
    """Install the ComfyUI APIs used by both runtime adapters."""

    state = FakeComfyState()
    comfy_module = ModuleType("comfy")
    comfy_sd = ModuleType("comfy.sd")
    comfy_sd.CLIPType = FakeCLIPType  # type: ignore[attr-defined]

    def load_diffusion_model(
        path: str,
        model_options: dict[str, object],
    ) -> object:
        """Record standalone diffusion loading."""

        state.diffusion_calls.append((path, model_options))
        return "model"

    def load_clip(
        ckpt_paths: list[str],
        embedding_directory: list[str],
        clip_type: FakeCLIPType,
        model_options: dict[str, object],
    ) -> object:
        """Record text-encoder loading."""

        state.clip_calls.append(
            {
                "paths": ckpt_paths,
                "embedding_directory": embedding_directory,
                "clip_type": clip_type,
                "model_options": model_options,
            }
        )
        return "clip"

    comfy_sd.load_diffusion_model = load_diffusion_model  # type: ignore[attr-defined]
    comfy_sd.load_clip = load_clip  # type: ignore[attr-defined]
    comfy_module.sd = comfy_sd  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)
    return state


def _artifact() -> AutoModelArtifact:
    """Return a small trusted artifact for component orchestration tests."""

    return AutoModelArtifact(
        cache_id="test_encoder",
        filename="auto.safetensors",
        folder_name="text_encoders",
        canonical_subfolder="flux",
        source_url="https://example.invalid/auto.safetensors",
        source_repo="example/auto",
        description="test encoder",
        sha256="0" * 64,
    )
