# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for settings-aware loader model choices."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from simple_syrup.runtime.model_choices import (
    NO_LOCAL_GROUNDING_DINO_MODELS,
    NO_LOCAL_SAM_MODELS,
    NO_LOCAL_VITMATTE_MODELS,
    NO_LOCAL_WD14_TAGGER_MODELS,
    ModelChoiceService,
)
from simple_syrup.runtime.settings import SimpleSyrupSettings


class FakeSettingsRepository:
    """Settings repository double for model choice tests."""

    def __init__(self, show_downloadable_models: bool) -> None:
        """Store the setting value returned by `load()`."""

        self._settings = SimpleSyrupSettings(show_downloadable_models)

    def load(self) -> SimpleSyrupSettings:
        """Return the configured settings."""

        return self._settings


def test_downloadable_mode_includes_catalog_entries(tmp_path: Path) -> None:
    """Downloadable mode shows known catalog entries even when files are absent."""

    service = ModelChoiceService(
        FakeSettingsRepository(show_downloadable_models=True),
        fake_folder_paths(tmp_path),
    )

    assert "sam_vit_b (375MB)" in service.sam_choices()
    assert "GroundingDINO_SwinT_OGC (694MB)" in service.grounding_dino_choices()
    assert "vitmatte-small-composition-1k" in service.vitmatte_choices()
    assert "wd-eva02-large-tagger-v3" in service.wd14_tagger_choices()


def test_local_only_mode_returns_sentinels_when_no_models_exist(
    tmp_path: Path,
) -> None:
    """Local-only mode shows clear sentinel values when folders are empty."""

    service = local_only_service(tmp_path)

    assert service.sam_choices() == [NO_LOCAL_SAM_MODELS]
    assert service.grounding_dino_choices() == [NO_LOCAL_GROUNDING_DINO_MODELS]
    assert service.vitmatte_choices() == [NO_LOCAL_VITMATTE_MODELS]
    assert service.wd14_tagger_choices() == [NO_LOCAL_WD14_TAGGER_MODELS]


def test_sam_local_only_lists_installed_catalog_artifacts(tmp_path: Path) -> None:
    """SAM local-only mode lists installed known checkpoint files."""

    (tmp_path / "models" / "sams").mkdir(parents=True)
    (tmp_path / "models" / "sams" / "sam_vit_b_01ec64.pth").write_bytes(b"sam")

    choices = local_only_service(tmp_path).sam_choices()

    assert choices == ["sam_vit_b (375MB)"]


def test_grounding_dino_local_only_requires_complete_artifacts(
    tmp_path: Path,
) -> None:
    """GroundingDINO local-only mode excludes partial config/checkpoint pairs."""

    model_dir = tmp_path / "models" / "grounding-dino"
    model_dir.mkdir(parents=True)
    (model_dir / "GroundingDINO_SwinT_OGC.cfg.py").write_text("", encoding="utf-8")
    (model_dir / "groundingdino_swint_ogc.pth").write_bytes(b"dino")
    (model_dir / "GroundingDINO_SwinB.cfg.py").write_text("", encoding="utf-8")

    choices = local_only_service(tmp_path).grounding_dino_choices()

    assert choices == ["GroundingDINO_SwinT_OGC (694MB)"]


def test_vitmatte_local_only_lists_valid_canonical_directory(
    tmp_path: Path,
) -> None:
    """ViTMatte local-only mode accepts canonical SimpleSyrup directories."""

    create_vitmatte_snapshot(
        tmp_path / "models" / "vitmatte" / "vitmatte-small-composition-1k"
    )

    choices = local_only_service(tmp_path).vitmatte_choices()

    assert choices == ["vitmatte-small-composition-1k"]


def test_vitmatte_local_only_lists_layerstyle_directory(tmp_path: Path) -> None:
    """ViTMatte local-only mode accepts LayerStyle-compatible directories."""

    create_vitmatte_snapshot(tmp_path / "models" / "vitmatte-base-composition-1k")

    choices = local_only_service(tmp_path).vitmatte_choices()

    assert choices == ["vitmatte-base-composition-1k"]


def test_wd14_local_only_requires_complete_artifacts(tmp_path: Path) -> None:
    """WD14 local-only mode excludes partial ONNX/CSV pairs."""

    model_dir = tmp_path / "models" / "wd14_tagger"
    model_dir.mkdir(parents=True)
    (model_dir / "wd-eva02-large-tagger-v3.onnx").write_bytes(b"onnx")
    (model_dir / "wd-eva02-large-tagger-v3.csv").write_text(
        "name,tag,category\n",
        encoding="utf-8",
    )
    (model_dir / "wd-vit-tagger-v3.onnx").write_bytes(b"onnx")

    choices = local_only_service(tmp_path).wd14_tagger_choices()

    assert choices == ["wd-eva02-large-tagger-v3"]


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        (NO_LOCAL_SAM_MODELS, "No local SAM models are available"),
        (NO_LOCAL_GROUNDING_DINO_MODELS, "No local GroundingDINO models are available"),
        (NO_LOCAL_VITMATTE_MODELS, "No local ViTMatte models are available"),
        (NO_LOCAL_WD14_TAGGER_MODELS, "No local WD14 tagger models are available"),
    ],
)
def test_sentinel_validation_raises_actionable_errors(
    tmp_path: Path,
    selection: str,
    message: str,
) -> None:
    """Sentinel dropdown selections fail before loader resolution work."""

    service = local_only_service(tmp_path)

    with pytest.raises(ValueError, match=message):
        service.reject_sentinel(selection)


def local_only_service(tmp_path: Path) -> ModelChoiceService:
    """Create a local-only choice service rooted in a temporary model folder."""

    return ModelChoiceService(
        FakeSettingsRepository(show_downloadable_models=False),
        fake_folder_paths(tmp_path),
    )


def fake_folder_paths(tmp_path: Path) -> ModuleType:
    """Create a minimal fake Comfy folder_paths module."""

    module = ModuleType("folder_paths")
    module.models_dir = str(tmp_path / "models")  # type: ignore[attr-defined]
    module.folder_names_and_paths = {}  # type: ignore[attr-defined]
    return module


def create_vitmatte_snapshot(path: Path) -> None:
    """Create the minimal file set required for a valid ViTMatte directory."""

    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"vitmatte")
