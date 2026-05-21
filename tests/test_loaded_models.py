# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for loaded model containers."""

from __future__ import annotations

from pathlib import Path

import pytest

from simple_syrup.runtime.loaded_models import (
    LoadedGroundingDINOModel,
    LoadedSAMModel,
    LoadedViTMatteModel,
    LoadedWD14Tagger,
    unwrap_grounding_dino_model,
    unwrap_sam_model,
    unwrap_vitmatte_model,
    unwrap_wd14_tagger,
)
from simple_syrup.runtime.wd14_tagger import FloatArray, WD14TagRecord


def test_loaded_sam_model_unwraps_underlying_model() -> None:
    """SAM containers preserve model metadata and unwrap cleanly."""

    model = object()
    loaded = LoadedSAMModel(model=model, source="local", model_id="sam_vit_b")

    assert loaded.source == "local"
    assert unwrap_sam_model(loaded) is model


def test_loaded_grounding_dino_model_unwraps_underlying_model(tmp_path: Path) -> None:
    """GroundingDINO containers preserve text encoder metadata."""

    model = object()
    loaded = LoadedGroundingDINOModel(
        model=model,
        text_encoder_path=tmp_path / "bert",
        source="local",
        model_id="groundingdino_swint_ogc",
    )

    assert loaded.text_encoder_path == tmp_path / "bert"
    assert unwrap_grounding_dino_model(loaded) is model


def test_loaded_vitmatte_model_preserves_model_and_processor(tmp_path: Path) -> None:
    """ViTMatte containers preserve model, processor, and source path metadata."""

    model = object()
    processor = object()
    loaded = LoadedViTMatteModel(
        model=model,
        processor=processor,
        source="local",
        model_id="vitmatte-small-composition-1k",
        model_path=tmp_path / "vitmatte-small-composition-1k",
    )

    assert loaded.processor is processor
    assert unwrap_vitmatte_model(loaded) is loaded


def test_loaded_wd14_tagger_preserves_runtime_metadata(tmp_path: Path) -> None:
    """WD14 containers preserve loaded session, tags, paths, and provider metadata."""

    session = _FakeWD14Session()
    tags = (WD14TagRecord("blue_hair", "0"),)
    loaded = LoadedWD14Tagger(
        model_id="wd-eva02-large-tagger-v3",
        source="local",
        onnx_path=tmp_path / "model.onnx",
        csv_path=tmp_path / "tags.csv",
        providers=("CPUExecutionProvider",),
        session=session,
        tags=tags,
    )

    assert loaded.session is session
    assert loaded.tags == tags
    assert unwrap_wd14_tagger(loaded) is loaded


def test_unwrap_wd14_tagger_rejects_incompatible_object() -> None:
    """WD14 unwrap errors name the loader required for compatible objects."""

    with pytest.raises(TypeError, match="Load WD14 Tagger"):
        unwrap_wd14_tagger(object())


class _FakeWD14Session:
    """Minimal WD14 session test double."""

    def get_inputs(self) -> list[object]:
        """Return no fake inputs."""

        return []

    def get_outputs(self) -> list[object]:
        """Return no fake outputs."""

        return []

    def run(
        self, output_names: list[str], feeds: dict[str, FloatArray]
    ) -> list[object]:
        """Return no fake outputs."""

        _ = output_names, feeds
        return []
