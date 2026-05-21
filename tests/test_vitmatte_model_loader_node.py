# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ViTMatte Model Loader node declaration."""

from __future__ import annotations

from typing import Any

import pytest

from simple_syrup.nodes.vitmatte_model_loader import ViTMatteModelLoader
from simple_syrup.runtime.model_choices import NO_LOCAL_VITMATTE_MODELS


def test_vitmatte_model_loader_contract() -> None:
    """ViTMatte loader exposes the conventional VITMATTE_MODEL socket."""

    assert ViTMatteModelLoader.RETURN_TYPES == ("VITMATTE_MODEL",)
    assert ViTMatteModelLoader.RETURN_NAMES == ("vitmatte_model",)
    assert ViTMatteModelLoader.FUNCTION == "load_model"
    assert ViTMatteModelLoader.CATEGORY == "SimpleSyrup/Masking"


def test_vitmatte_model_loader_declares_expected_inputs() -> None:
    """ViTMatte loader inputs are asset-only and deterministic."""

    input_types: dict[str, dict[str, tuple[Any, ...]]] = (
        ViTMatteModelLoader.INPUT_TYPES()
    )
    required = input_types["required"]

    assert set(required) == {"vitmatte_model"}
    assert required["vitmatte_model"][0] == [
        "vitmatte-small-composition-1k",
        "vitmatte-base-composition-1k",
    ]


def test_vitmatte_model_loader_uses_settings_aware_choices() -> None:
    """ViTMatte loader dropdown choices come from the choice service."""

    class FakeChoices:
        """Choice service double for INPUT_TYPES."""

        def vitmatte_choices(self) -> list[str]:
            """Return local-only choices."""

            return ["vitmatte-base-composition-1k"]

    original = ViTMatteModelLoader._choices
    ViTMatteModelLoader._choices = FakeChoices()  # type: ignore[assignment]
    try:
        required = ViTMatteModelLoader.INPUT_TYPES()["required"]
    finally:
        ViTMatteModelLoader._choices = original

    assert required["vitmatte_model"][0] == ["vitmatte-base-composition-1k"]
    assert required["vitmatte_model"][1]["default"] == "vitmatte-base-composition-1k"


def test_vitmatte_model_loader_delegates_to_service() -> None:
    """Node execution delegates to the loader service."""

    expected = object()

    class FakeService:
        """Service double for node delegation."""

        def load_model(self, **kwargs: object) -> object:
            """Return a fixed model object."""

            return expected

    node = ViTMatteModelLoader()
    original = ViTMatteModelLoader._service
    ViTMatteModelLoader._service = FakeService()  # type: ignore[assignment]
    try:
        result = node.load_model(
            vitmatte_model="vitmatte-small-composition-1k",
        )
    finally:
        ViTMatteModelLoader._service = original

    assert result == (expected,)


def test_vitmatte_model_loader_rejects_sentinel_selection() -> None:
    """ViTMatte loader rejects no-local-model sentinel selections."""

    node = ViTMatteModelLoader()

    with pytest.raises(ValueError, match="No local ViTMatte models are available"):
        node.load_model(vitmatte_model=NO_LOCAL_VITMATTE_MODELS)


def test_vitmatte_model_loader_always_allows_downloads_for_selected_models() -> None:
    """Selected downloadable models are resolved with internal download enabled."""

    class FakeService:
        """Service double that records node download policy."""

        def __init__(self) -> None:
            """Create a recording service double."""

            self.kwargs: dict[str, object] | None = None

        def load_model(self, **kwargs: object) -> object:
            """Record call arguments and return a fixed model object."""

            self.kwargs = kwargs
            return object()

    fake_service = FakeService()
    original = ViTMatteModelLoader._service
    ViTMatteModelLoader._service = fake_service  # type: ignore[assignment]
    try:
        ViTMatteModelLoader().load_model(
            vitmatte_model="vitmatte-small-composition-1k",
        )
    finally:
        ViTMatteModelLoader._service = original

    assert fake_service.kwargs is not None
    assert fake_service.kwargs["auto_download"] is True
