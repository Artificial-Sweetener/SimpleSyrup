# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Load Ultralytics Model node contract."""

from __future__ import annotations

from typing import Any, cast

import pytest

from simple_syrup.nodes.load_ultralytics_model import LoadUltralyticsModel
from simple_syrup.runtime.ultralytics_loader import LoadedUltralyticsDetector


def test_load_ultralytics_model_node_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load Ultralytics Model exposes the planned return types and category."""

    monkeypatch.setattr(LoadUltralyticsModel, "service_class", _FakeLoaderService)

    inputs = LoadUltralyticsModel.INPUT_TYPES()

    assert LoadUltralyticsModel.RETURN_TYPES == (
        "DETECTOR_MODEL",
        "BBOX_DETECTOR",
        "SEGM_DETECTOR",
    )
    assert LoadUltralyticsModel.CATEGORY == "SimpleSyrup/Detection"
    assert inputs["required"]["model_name"][0] == ["model.pt"]


def test_load_ultralytics_model_node_returns_loaded_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load Ultralytics Model forwards native and compatibility outputs."""

    monkeypatch.setattr(LoadUltralyticsModel, "service_class", _FakeLoaderService)

    assert LoadUltralyticsModel().load("model.pt") == ("native", "bbox", "segm")


class _FakeLoaderService:
    """Fake loader service for node tests."""

    def model_choices(self) -> list[str]:
        """Return deterministic dropdown choices."""

        return ["model.pt"]

    def load(self, model_name: str) -> LoadedUltralyticsDetector:
        """Return deterministic loaded outputs."""

        assert model_name == "model.pt"
        return LoadedUltralyticsDetector(cast(Any, "native"), "bbox", "segm")
