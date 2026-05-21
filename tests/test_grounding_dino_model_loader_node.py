# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the GroundingDINO Model Loader node declaration."""

from __future__ import annotations

from typing import Any

import pytest

from simple_syrup.nodes.grounding_dino_model_loader import GroundingDINOModelLoader
from simple_syrup.runtime.grounding_dino_loader import (
    TEXT_ENCODER_AUTO,
    TEXT_ENCODER_COMFY,
    TEXT_ENCODER_LAYERSTYLE,
)
from simple_syrup.runtime.model_choices import NO_LOCAL_GROUNDING_DINO_MODELS


def test_grounding_dino_model_loader_contract() -> None:
    """GroundingDINO loader exposes the SimpleSyrup model socket."""

    assert GroundingDINOModelLoader.RETURN_TYPES == ("GROUNDING_DINO_MODEL",)
    assert GroundingDINOModelLoader.RETURN_NAMES == ("grounding_dino_model",)
    assert GroundingDINOModelLoader.FUNCTION == "load_model"
    assert GroundingDINOModelLoader.CATEGORY == "SimpleSyrup/Masking"


def test_grounding_dino_model_loader_declares_expected_inputs() -> None:
    """GroundingDINO loader makes text encoder selection explicit."""

    input_types: dict[str, dict[str, tuple[Any, ...]]] = (
        GroundingDINOModelLoader.INPUT_TYPES()
    )
    required = input_types["required"]

    assert set(required) == {
        "grounding_dino_model",
        "text_encoder",
    }
    assert "GroundingDINO_SwinT_OGC (694MB)" in required["grounding_dino_model"][0]
    assert TEXT_ENCODER_LAYERSTYLE in required["text_encoder"][0]
    assert TEXT_ENCODER_COMFY in required["text_encoder"][0]
    assert TEXT_ENCODER_AUTO in required["text_encoder"][0]


def test_grounding_dino_model_loader_uses_settings_aware_choices() -> None:
    """GroundingDINO loader dropdown choices come from the choice service."""

    class FakeChoices:
        """Choice service double for INPUT_TYPES."""

        def grounding_dino_choices(self) -> list[str]:
            """Return local-only choices."""

            return ["GroundingDINO_SwinB (938MB)"]

    original = GroundingDINOModelLoader._choices
    GroundingDINOModelLoader._choices = FakeChoices()  # type: ignore[assignment]
    try:
        required = GroundingDINOModelLoader.INPUT_TYPES()["required"]
    finally:
        GroundingDINOModelLoader._choices = original

    assert required["grounding_dino_model"][0] == ["GroundingDINO_SwinB (938MB)"]
    assert (
        required["grounding_dino_model"][1]["default"] == "GroundingDINO_SwinB (938MB)"
    )


def test_grounding_dino_model_loader_delegates_to_service() -> None:
    """Node execution delegates to the loader service."""

    expected = object()

    class FakeService:
        """Service double for node delegation."""

        def load_model(self, **kwargs: object) -> object:
            """Return a fixed model object."""

            return expected

    node = GroundingDINOModelLoader()
    original = GroundingDINOModelLoader._service
    GroundingDINOModelLoader._service = FakeService()  # type: ignore[assignment]
    try:
        result = node.load_model(
            grounding_dino_model="GroundingDINO_SwinT_OGC (694MB)",
            text_encoder=TEXT_ENCODER_AUTO,
        )
    finally:
        GroundingDINOModelLoader._service = original

    assert result == (expected,)


def test_grounding_dino_model_loader_rejects_sentinel_selection() -> None:
    """GroundingDINO loader rejects no-local-model sentinel selections."""

    node = GroundingDINOModelLoader()

    with pytest.raises(
        ValueError,
        match="No local GroundingDINO models are available",
    ):
        node.load_model(
            grounding_dino_model=NO_LOCAL_GROUNDING_DINO_MODELS,
            text_encoder=TEXT_ENCODER_AUTO,
        )


def test_grounding_dino_model_loader_always_allows_downloads_for_selected_models() -> (
    None
):
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
    original = GroundingDINOModelLoader._service
    GroundingDINOModelLoader._service = fake_service  # type: ignore[assignment]
    try:
        GroundingDINOModelLoader().load_model(
            grounding_dino_model="GroundingDINO_SwinT_OGC (694MB)",
            text_encoder=TEXT_ENCODER_AUTO,
        )
    finally:
        GroundingDINOModelLoader._service = original

    assert fake_service.kwargs is not None
    assert fake_service.kwargs["auto_download"] is True
