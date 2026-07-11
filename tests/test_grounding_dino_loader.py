# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for GroundingDINO loader runtime service."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from simple_syrup.runtime.grounding_dino_loader import (
    GROUNDING_DINO_RUNTIME_PACKAGE,
    TEXT_ENCODER_AUTO,
    TEXT_ENCODER_COMFY,
    TEXT_ENCODER_LAYERSTYLE,
    GroundingDINOLoaderService,
    GroundingDINOModelCacheKey,
)
from simple_syrup.runtime.loaded_models import LoadedGroundingDINOModel
from test_helpers import FakeFolderPaths


@dataclass
class _RecordingPhaseProgress:
    """Record semantic model-load phases emitted by the runtime service."""

    phases: list[str] = field(default_factory=list)

    def advance(self, phase: str) -> None:
        """Record one phase transition."""

        self.phases.append(phase)


def test_grounding_dino_loader_resolves_explicit_layerstyle_bert(
    tmp_path: Path,
) -> None:
    """Explicit LayerStyle BERT mode uses models/bert-base-uncased."""

    _write_bert(tmp_path / "bert-base-uncased")
    resolved = GroundingDINOLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path)
    ).resolve_text_encoder(TEXT_ENCODER_LAYERSTYLE, auto_download=False)

    assert resolved.path == tmp_path / "bert-base-uncased"


def test_grounding_dino_loader_resolves_explicit_text_encoder_bert(
    tmp_path: Path,
) -> None:
    """Explicit ComfyUI text encoder mode uses text_encoders/bert."""

    _write_bert(tmp_path / "text_encoders" / "bert")
    resolved = GroundingDINOLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path)
    ).resolve_text_encoder(TEXT_ENCODER_COMFY, auto_download=False)

    assert resolved.path == tmp_path / "text_encoders" / "bert"


def test_grounding_dino_loader_auto_prefers_layerstyle_bert(tmp_path: Path) -> None:
    """Auto text encoder mode preserves the preferred local order."""

    _write_bert(tmp_path / "bert-base-uncased")
    _write_bert(tmp_path / "text_encoders" / "bert")
    resolved = GroundingDINOLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path)
    ).resolve_text_encoder(TEXT_ENCODER_AUTO, auto_download=False)

    assert resolved.path == tmp_path / "bert-base-uncased"


def test_grounding_dino_loader_explicit_missing_bert_does_not_download(
    tmp_path: Path,
) -> None:
    """Explicit text encoder modes fail instead of silently using another path."""

    with pytest.raises(FileNotFoundError, match="text_encoders/bert"):
        GroundingDINOLoaderService(
            folder_paths_module=FakeFolderPaths(tmp_path)
        ).resolve_text_encoder(TEXT_ENCODER_COMFY, auto_download=True)


def test_grounding_dino_loader_uses_process_cache_for_identical_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical GroundingDINO loads reuse the same loaded container."""

    state = _install_fake_grounding_dino(monkeypatch)
    _write_grounding_dino_artifacts(tmp_path)
    _write_bert(tmp_path / "bert-base-uncased")
    cache: dict[GroundingDINOModelCacheKey, LoadedGroundingDINOModel] = {}
    service = GroundingDINOLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        cache=cache,
    )

    first = service.load_model(
        "GroundingDINO_SwinT_OGC (694MB)",
        TEXT_ENCODER_LAYERSTYLE,
        auto_download=True,
    )
    second = service.load_model(
        "GroundingDINO_SwinT_OGC (694MB)",
        TEXT_ENCODER_LAYERSTYLE,
        auto_download=True,
    )

    assert second is first
    assert first.managed_model is not None
    assert state.config_paths == [
        str(tmp_path / "grounding-dino" / "GroundingDINO_SwinT_OGC.cfg.py")
    ]
    assert state.checkpoint_paths == [
        str(tmp_path / "grounding-dino" / "groundingdino_swint_ogc.pth")
    ]
    assert len(cache) == 1


def test_grounding_dino_loader_reports_cache_miss_and_hit_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GroundingDINO loads should expose construction and cache-hit boundaries."""

    _install_fake_grounding_dino(monkeypatch)
    _write_grounding_dino_artifacts(tmp_path)
    _write_bert(tmp_path / "bert-base-uncased")
    service = GroundingDINOLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        cache={},
    )
    cache_miss_progress = _RecordingPhaseProgress()
    cache_hit_progress = _RecordingPhaseProgress()

    service.load_model(
        "GroundingDINO_SwinT_OGC (694MB)",
        TEXT_ENCODER_LAYERSTYLE,
        auto_download=True,
        phase_progress=cache_miss_progress,
    )
    service.load_model(
        "GroundingDINO_SwinT_OGC (694MB)",
        TEXT_ENCODER_LAYERSTYLE,
        auto_download=True,
        phase_progress=cache_hit_progress,
    )

    assert cache_miss_progress.phases == [
        "resolving_artifacts",
        "constructing_model",
        "loading_checkpoint",
        "loading_state_dict",
        "registering_device_management",
        "completed",
    ]
    assert cache_hit_progress.phases == [
        "resolving_artifacts",
        "cache_hit",
        "completed",
    ]


def test_grounding_dino_loader_invalidates_import_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vendored runtime imports refresh caches for already-running Comfy processes."""

    state = _install_fake_grounding_dino(monkeypatch)
    invalidations = 0

    def invalidate_caches() -> None:
        nonlocal invalidations
        invalidations += 1

    monkeypatch.setattr(importlib, "invalidate_caches", invalidate_caches)
    _write_grounding_dino_artifacts(tmp_path)
    _write_bert(tmp_path / "bert-base-uncased")

    GroundingDINOLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        cache={},
    ).load_model(
        "GroundingDINO_SwinT_OGC (694MB)",
        TEXT_ENCODER_LAYERSTYLE,
        auto_download=True,
    )

    assert state.build_calls == 1
    assert invalidations == 1


def test_grounding_dino_loader_cache_separates_text_encoder_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different BERT directories produce separate GroundingDINO instances."""

    state = _install_fake_grounding_dino(monkeypatch)
    _write_grounding_dino_artifacts(tmp_path)
    _write_bert(tmp_path / "bert-base-uncased")
    _write_bert(tmp_path / "text_encoders" / "bert")
    cache: dict[GroundingDINOModelCacheKey, LoadedGroundingDINOModel] = {}
    service = GroundingDINOLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        cache=cache,
    )

    first = service.load_model(
        "GroundingDINO_SwinT_OGC (694MB)",
        TEXT_ENCODER_LAYERSTYLE,
        auto_download=True,
    )
    second = service.load_model(
        "GroundingDINO_SwinT_OGC (694MB)",
        TEXT_ENCODER_COMFY,
        auto_download=True,
    )

    assert second is not first
    assert state.text_encoder_paths == [
        str(tmp_path / "bert-base-uncased"),
        str(tmp_path / "text_encoders" / "bert"),
    ]
    assert len(cache) == 2


def test_grounding_dino_loader_does_not_cache_failed_model_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed GroundingDINO build leaves the cache empty for retry."""

    state = _install_fake_grounding_dino(monkeypatch, fail_once=True)
    _write_grounding_dino_artifacts(tmp_path)
    _write_bert(tmp_path / "bert-base-uncased")
    cache: dict[GroundingDINOModelCacheKey, LoadedGroundingDINOModel] = {}
    service = GroundingDINOLoaderService(
        folder_paths_module=FakeFolderPaths(tmp_path),
        cache=cache,
    )

    phase_progress = _RecordingPhaseProgress()
    with pytest.raises(RuntimeError, match="GroundingDINO failed"):
        service.load_model(
            "GroundingDINO_SwinT_OGC (694MB)",
            TEXT_ENCODER_LAYERSTYLE,
            auto_download=True,
            phase_progress=phase_progress,
        )

    assert phase_progress.phases == [
        "resolving_artifacts",
        "constructing_model",
        "failed",
    ]

    loaded = service.load_model(
        "GroundingDINO_SwinT_OGC (694MB)",
        TEXT_ENCODER_LAYERSTYLE,
        auto_download=True,
    )

    assert isinstance(loaded, LoadedGroundingDINOModel)
    assert state.build_calls == 2
    assert len(cache) == 1


def _write_bert(path: Path) -> None:
    """Write a minimal valid BERT directory."""

    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


def _write_grounding_dino_artifacts(tmp_path: Path) -> None:
    """Write a minimal GroundingDINO config and checkpoint pair."""

    model_dir = tmp_path / "grounding-dino"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "GroundingDINO_SwinT_OGC.cfg.py").write_text(
        "# config\n",
        encoding="utf-8",
    )
    (model_dir / "groundingdino_swint_ogc.pth").write_bytes(b"checkpoint")


@dataclass
class _FakeGroundingDINOState:
    """Record fake GroundingDINO runtime construction."""

    config_paths: list[str] = field(default_factory=list)
    checkpoint_paths: list[str] = field(default_factory=list)
    text_encoder_paths: list[str] = field(default_factory=list)
    build_calls: int = 0
    fail_once: bool = False


class _FakeArgs:
    """Small GroundingDINO config object fake."""

    def __init__(self) -> None:
        """Create args that trigger text encoder path substitution."""

        self.text_encoder_type = "bert-base-uncased"


class _FakeSLConfig:
    """Fake GroundingDINO SLConfig factory."""

    state: _FakeGroundingDINOState

    @classmethod
    def fromfile(cls, path: str) -> _FakeArgs:
        """Record config loading and return mutable fake args."""

        cls.state.config_paths.append(path)
        return _FakeArgs()


class _FakeGroundingDINOModel:
    """Minimal GroundingDINO model fake."""

    def __init__(self, state: _FakeGroundingDINOState, text_encoder_path: str) -> None:
        """Create a model fake that records selected BERT path."""

        self.state = state
        self.text_encoder_path = text_encoder_path
        self.model_name = ""
        self.load_calls = 0
        self.eval_calls = 0

    def load_state_dict(self, state_dict: dict[str, object], strict: bool) -> None:
        """Record model weight loading."""

        _ = state_dict, strict
        self.load_calls += 1

    def eval(self) -> None:
        """Record eval mode selection."""

        self.eval_calls += 1


def _install_fake_grounding_dino(
    monkeypatch: pytest.MonkeyPatch,
    fail_once: bool = False,
) -> _FakeGroundingDINOState:
    """Install fake GroundingDINO and torch load boundaries."""

    state = _FakeGroundingDINOState(fail_once=fail_once)
    _FakeSLConfig.state = state

    groundingdino = ModuleType(GROUNDING_DINO_RUNTIME_PACKAGE)
    util = ModuleType(f"{GROUNDING_DINO_RUNTIME_PACKAGE}.util")
    slconfig = ModuleType(f"{GROUNDING_DINO_RUNTIME_PACKAGE}.util.slconfig")
    utils = ModuleType(f"{GROUNDING_DINO_RUNTIME_PACKAGE}.util.utils")
    models = ModuleType(f"{GROUNDING_DINO_RUNTIME_PACKAGE}.models")

    slconfig.SLConfig = _FakeSLConfig  # type: ignore[attr-defined]
    utils.clean_state_dict = lambda state_dict: state_dict  # type: ignore[attr-defined]

    def build_model(args: Any) -> _FakeGroundingDINOModel:
        """Record model construction and optionally fail once."""

        state.build_calls += 1
        state.text_encoder_paths.append(str(args.text_encoder_type))
        if state.fail_once:
            state.fail_once = False
            raise RuntimeError("GroundingDINO failed")
        return _FakeGroundingDINOModel(state, str(args.text_encoder_type))

    def load_checkpoint(path: str, map_location: str) -> dict[str, dict[str, object]]:
        """Record checkpoint loading and return a fake checkpoint."""

        assert map_location == "cpu"
        state.checkpoint_paths.append(path)
        return {"model": {"weight": object()}}

    models.build_model = build_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, GROUNDING_DINO_RUNTIME_PACKAGE, groundingdino)
    monkeypatch.setitem(sys.modules, f"{GROUNDING_DINO_RUNTIME_PACKAGE}.util", util)
    monkeypatch.setitem(
        sys.modules,
        f"{GROUNDING_DINO_RUNTIME_PACKAGE}.util.slconfig",
        slconfig,
    )
    monkeypatch.setitem(
        sys.modules,
        f"{GROUNDING_DINO_RUNTIME_PACKAGE}.util.utils",
        utils,
    )
    monkeypatch.setitem(sys.modules, f"{GROUNDING_DINO_RUNTIME_PACKAGE}.models", models)
    monkeypatch.setattr(torch, "load", load_checkpoint)
    return state
