# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Load WD14 Tagger node declaration."""

from __future__ import annotations

from typing import Any

import pytest

from simple_syrup.nodes.wd14_tagger_loader import WD14TaggerLoader
from simple_syrup.runtime.model_choices import NO_LOCAL_WD14_TAGGER_MODELS


def test_wd14_tagger_loader_contract() -> None:
    """WD14 loader exposes the conventional WD14_TAGGER socket."""

    assert WD14TaggerLoader.RETURN_TYPES == ("WD14_TAGGER",)
    assert WD14TaggerLoader.RETURN_NAMES == ("wd14_tagger",)
    assert WD14TaggerLoader.FUNCTION == "load_model"
    assert WD14TaggerLoader.CATEGORY == "SimpleSyrup/Tagging"


def test_wd14_tagger_loader_declares_expected_inputs() -> None:
    """WD14 loader inputs are asset-only and deterministic."""

    input_types: dict[str, dict[str, tuple[Any, ...]]] = WD14TaggerLoader.INPUT_TYPES()
    required = input_types["required"]

    assert set(required) == {"wd14_model"}
    assert required["wd14_model"][0][0] == "wd-eva02-large-tagger-v3"
    assert required["wd14_model"][1]["default"] == "wd-eva02-large-tagger-v3"


def test_wd14_tagger_loader_uses_settings_aware_choices() -> None:
    """WD14 loader dropdown choices come from the choice service."""

    class FakeChoices:
        """Choice service double for INPUT_TYPES."""

        def wd14_tagger_choices(self) -> list[str]:
            """Return local-only choices."""

            return ["wd-vit-tagger-v3"]

    original = WD14TaggerLoader._choices
    WD14TaggerLoader._choices = FakeChoices()  # type: ignore[assignment]
    try:
        required = WD14TaggerLoader.INPUT_TYPES()["required"]
    finally:
        WD14TaggerLoader._choices = original

    assert required["wd14_model"][0] == ["wd-vit-tagger-v3"]
    assert required["wd14_model"][1]["default"] == "wd-vit-tagger-v3"


def test_wd14_tagger_loader_delegates_to_service() -> None:
    """Node execution delegates to the loader service."""

    expected = object()

    class FakeService:
        """Service double for node delegation."""

        def load_model(self, **kwargs: object) -> object:
            """Return a fixed model object."""

            return expected

    node = WD14TaggerLoader()
    original = WD14TaggerLoader._service
    WD14TaggerLoader._service = FakeService()  # type: ignore[assignment]
    try:
        result = node.load_model(wd14_model="wd-eva02-large-tagger-v3")
    finally:
        WD14TaggerLoader._service = original

    assert result == (expected,)


def test_wd14_tagger_loader_rejects_sentinel_selection() -> None:
    """WD14 loader rejects no-local-model sentinel selections."""

    node = WD14TaggerLoader()

    with pytest.raises(ValueError, match="No local WD14 tagger models are available"):
        node.load_model(wd14_model=NO_LOCAL_WD14_TAGGER_MODELS)


def test_wd14_tagger_loader_always_allows_downloads_for_selected_models() -> None:
    """Selected downloadable WD14 models are resolved with internal download enabled."""

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
    original = WD14TaggerLoader._service
    WD14TaggerLoader._service = fake_service  # type: ignore[assignment]
    try:
        WD14TaggerLoader().load_model(wd14_model="wd-eva02-large-tagger-v3")
    finally:
        WD14TaggerLoader._service = original

    assert fake_service.kwargs is not None
    assert fake_service.kwargs["auto_download"] is True
