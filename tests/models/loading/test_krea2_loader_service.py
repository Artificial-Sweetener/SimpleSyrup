# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Krea 2 component selection and runtime validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

import simple_syrup.services.krea2_loader_service as service_module
from simple_syrup.runtime.auto_model_artifact import AutoModelArtifact
from simple_syrup.runtime.auto_model_resolver import AutoModelResolution
from simple_syrup.runtime.diffusion_model_metadata import DiffusionModelMetadata
from simple_syrup.runtime.krea2_artifacts import (
    KREA2_AUTO_TEXT_ENCODER,
    KREA2_QWEN3_VL_4B_BF16,
    KREA2_QWEN3_VL_4B_FP8,
)
from simple_syrup.runtime.model_downloads import ProgressReporter
from simple_syrup.runtime.qwen_artifacts import QWEN_IMAGE_VAE
from simple_syrup.services.krea2_loader_service import Krea2LoaderService


class _FolderPaths(ModuleType):
    """Resolve deterministic manual text-encoder files."""

    def __init__(self, root: Path) -> None:
        """Store one fake models root."""

        super().__init__("folder_paths")
        self._root = root

    def get_full_path_or_raise(self, folder: str, name: str) -> str:
        """Return a deterministic path for a manual selection."""

        return str(self._root / folder / name)


class _DiffusionLoader:
    """Record diffusion requests and return a stable model object."""

    def __init__(self) -> None:
        """Create an empty request log."""

        self.calls: list[tuple[str, str]] = []
        self.model = object()

    def load(self, diffusion_model: str, weight_dtype: str) -> object:
        """Record one model load."""

        self.calls.append((diffusion_model, weight_dtype))
        return self.model


class _Inspector:
    """Return configured loaded-model metadata."""

    def __init__(self, image_model: str | None) -> None:
        """Store the image-model identifier to expose."""

        self.image_model = image_model
        self.calls: list[object] = []

    def inspect(self, model: object) -> DiffusionModelMetadata | None:
        """Record inspection and return configured metadata."""

        self.calls.append(model)
        if self.image_model is None:
            return None
        return DiffusionModelMetadata(self.image_model, None)


class _ClipTypeSupport:
    """Record or reject required Comfy CLIP types."""

    def __init__(self, error: Exception | None = None) -> None:
        """Configure an optional compatibility failure."""

        self.error = error
        self.calls: list[str] = []

    def require(self, clip_type_name: str) -> None:
        """Record the type and raise the configured failure."""

        self.calls.append(clip_type_name)
        if self.error is not None:
            raise self.error


@dataclass
class _Resolver:
    """Record automatic artifact resolutions."""

    root: Path
    calls: list[tuple[AutoModelArtifact, ProgressReporter | None]] = field(
        default_factory=list
    )

    def resolve(
        self,
        artifact: AutoModelArtifact,
        progress: ProgressReporter | None = None,
    ) -> AutoModelResolution:
        """Return a deterministic resolved path."""

        self.calls.append((artifact, progress))
        return AutoModelResolution(self.root / artifact.filename, "downloaded")


@dataclass
class _TextEncoderLoader:
    """Record Krea text-encoder adapter calls."""

    calls: list[tuple[tuple[Path, ...], str, str]] = field(default_factory=list)

    def load(
        self,
        paths: tuple[Path, ...],
        clip_type_name: str,
        device: str,
    ) -> object:
        """Record and return a stable CLIP value."""

        self.calls.append((paths, clip_type_name, device))
        return "clip"


@dataclass
class _VaeLoader:
    """Record manual VAE selections."""

    calls: list[str] = field(default_factory=list)

    def load_vae(self, vae_name: str) -> object:
        """Record and return a stable manual VAE value."""

        self.calls.append(vae_name)
        return "manual vae"


class _Progress:
    """Provide a progress object whose identity is asserted."""

    def start(self, label: str, total: int | None) -> None:
        """Accept a start update."""

    def advance(self, current: int, total: int | None) -> None:
        """Accept an absolute update."""

    def finish(self) -> None:
        """Accept completion."""


@pytest.mark.parametrize(
    ("selection", "artifact"),
    (
        (KREA2_AUTO_TEXT_ENCODER, KREA2_QWEN3_VL_4B_FP8),
        (KREA2_QWEN3_VL_4B_FP8.filename, KREA2_QWEN3_VL_4B_FP8),
        (KREA2_QWEN3_VL_4B_BF16.filename, KREA2_QWEN3_VL_4B_BF16),
    ),
)
def test_official_encoder_choices_resolve_downloadable_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
    artifact: AutoModelArtifact,
) -> None:
    """Auto, explicit FP8, and explicit BF16 share verified download behavior."""

    diffusion = _DiffusionLoader()
    resolver = _Resolver(tmp_path / "resolved")
    text_loader = _TextEncoderLoader()
    support = _ClipTypeSupport()
    progress = _Progress()
    loaded_vae_paths: list[Path] = []

    def load_automatic_vae(path: Path) -> object:
        """Record the resolved VAE path and return a stable object."""

        loaded_vae_paths.append(path)
        return "auto vae"

    monkeypatch.setattr(
        service_module,
        "load_vae_path",
        load_automatic_vae,
    )
    service = Krea2LoaderService(
        diffusion_loader=diffusion,
        text_encoder_loader=text_loader,
        model_inspector=_Inspector("krea2"),
        clip_type_support=support,
        resolver=resolver,
        vae_loader=_VaeLoader(),
        folder_paths_module=_FolderPaths(tmp_path),
    )

    result = service.load_models(
        "renamed-model.safetensors",
        "default",
        selection,
        "cpu",
        "auto",
        progress,
    )

    assert result == (diffusion.model, "clip", "auto vae")
    assert resolver.calls == [(artifact, progress), (QWEN_IMAGE_VAE, progress)]
    assert text_loader.calls == [
        ((tmp_path / "resolved" / artifact.filename,), "KREA2", "cpu")
    ]
    assert loaded_vae_paths == [tmp_path / "resolved" / QWEN_IMAGE_VAE.filename]
    assert support.calls == ["KREA2"]


def test_manual_components_use_comfy_folder_and_vae_boundaries(tmp_path: Path) -> None:
    """Manual escape hatches retain Krea validation and CLIP loading policy."""

    resolver = _Resolver(tmp_path / "resolved")
    text_loader = _TextEncoderLoader()
    vae_loader = _VaeLoader()
    service = Krea2LoaderService(
        diffusion_loader=_DiffusionLoader(),
        text_encoder_loader=text_loader,
        model_inspector=_Inspector("krea2"),
        clip_type_support=_ClipTypeSupport(),
        resolver=resolver,
        vae_loader=vae_loader,
        folder_paths_module=_FolderPaths(tmp_path),
    )

    result = service.load_models(
        "krea.safetensors",
        "fp8_e4m3fn_fast",
        "custom_qwen.safetensors",
        "default",
        "custom_vae.safetensors",
    )

    assert result[1:] == ("clip", "manual vae")
    assert resolver.calls == []
    assert text_loader.calls == [
        (
            (tmp_path / "text_encoders" / "custom_qwen.safetensors",),
            "KREA2",
            "default",
        )
    ]
    assert vae_loader.calls == ["custom_vae.safetensors"]


@pytest.mark.parametrize("image_model", [None, "flux", "flux2", "anima"])
def test_non_krea_models_fail_before_compatibility_checks_or_downloads(
    tmp_path: Path,
    image_model: str | None,
) -> None:
    """Structural validation prevents expensive support downloads for wrong models."""

    resolver = _Resolver(tmp_path)
    support = _ClipTypeSupport()
    service = Krea2LoaderService(
        diffusion_loader=_DiffusionLoader(),
        text_encoder_loader=_TextEncoderLoader(),
        model_inspector=_Inspector(image_model),
        clip_type_support=support,
        resolver=resolver,
        vae_loader=_VaeLoader(),
        folder_paths_module=_FolderPaths(tmp_path),
    )

    with pytest.raises(ValueError, match="recognizes as Krea 2"):
        service.load_models(
            "wrong.safetensors",
            "default",
            "auto",
            "default",
            "auto",
        )

    assert support.calls == []
    assert resolver.calls == []


def test_missing_host_krea_clip_type_fails_before_downloads(tmp_path: Path) -> None:
    """An outdated ComfyUI reports an actionable error without downloading files."""

    resolver = _Resolver(tmp_path)
    support = _ClipTypeSupport(RuntimeError("Update ComfyUI"))
    service = Krea2LoaderService(
        diffusion_loader=_DiffusionLoader(),
        text_encoder_loader=_TextEncoderLoader(),
        model_inspector=_Inspector("krea2"),
        clip_type_support=support,
        resolver=resolver,
        vae_loader=_VaeLoader(),
        folder_paths_module=_FolderPaths(tmp_path),
    )

    with pytest.raises(RuntimeError, match="Update ComfyUI"):
        service.load_models(
            "krea.safetensors",
            "default",
            "auto",
            "default",
            "auto",
        )

    assert resolver.calls == []
