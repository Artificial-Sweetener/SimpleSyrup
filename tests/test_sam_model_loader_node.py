# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the SAM Model Loader node declaration."""

from __future__ import annotations

from typing import Any

import pytest

from simple_syrup.nodes.sam_model_loader import SAMModelLoader
from simple_syrup.runtime.model_choices import NO_LOCAL_SAM_MODELS


def test_sam_model_loader_contract() -> None:
    """SAM loader exposes the conventional SAM_MODEL socket."""

    assert SAMModelLoader.RETURN_TYPES == ("SAM_MODEL",)
    assert SAMModelLoader.RETURN_NAMES == ("sam_model",)
    assert SAMModelLoader.FUNCTION == "load_model"
    assert SAMModelLoader.CATEGORY == "SimpleSyrup/Masking"


def test_sam_model_loader_declares_expected_inputs() -> None:
    """SAM loader inputs are deterministic and loader-owned."""

    input_types: dict[str, dict[str, tuple[Any, ...]]] = SAMModelLoader.INPUT_TYPES()
    required = input_types["required"]

    assert set(required) == {"sam_model"}
    assert "sam_vit_b (375MB)" in required["sam_model"][0]


def test_sam_model_loader_uses_settings_aware_choices() -> None:
    """SAM loader dropdown choices come from the model choice service."""

    class FakeChoices:
        """Choice service double for INPUT_TYPES."""

        def sam_choices(self) -> list[str]:
            """Return local-only choices."""

            return ["sam_vit_h (2.56GB)"]

    original = SAMModelLoader._choices
    SAMModelLoader._choices = FakeChoices()  # type: ignore[assignment]
    try:
        required = SAMModelLoader.INPUT_TYPES()["required"]
    finally:
        SAMModelLoader._choices = original

    assert required["sam_model"][0] == ["sam_vit_h (2.56GB)"]
    assert required["sam_model"][1]["default"] == "sam_vit_h (2.56GB)"


def test_sam_model_loader_delegates_to_service() -> None:
    """Node execution delegates to the loader service."""

    expected = object()

    class FakeService:
        """Service double for node delegation."""

        def load_model(self, **kwargs: object) -> object:
            """Return a fixed model object."""

            return expected

    node = SAMModelLoader()
    original = SAMModelLoader._service
    SAMModelLoader._service = FakeService()  # type: ignore[assignment]
    try:
        result = node.load_model(
            sam_model="sam_vit_b (375MB)",
        )
    finally:
        SAMModelLoader._service = original

    assert result == (expected,)


def test_sam_model_loader_rejects_sentinel_selection() -> None:
    """SAM loader rejects no-local-model sentinel selections."""

    node = SAMModelLoader()

    with pytest.raises(ValueError, match="No local SAM models are available"):
        node.load_model(sam_model=NO_LOCAL_SAM_MODELS)


def test_sam_model_loader_always_allows_downloads_for_selected_models() -> None:
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
    original = SAMModelLoader._service
    SAMModelLoader._service = fake_service  # type: ignore[assignment]
    try:
        SAMModelLoader().load_model(sam_model="sam_vit_b (375MB)")
    finally:
        SAMModelLoader._service = original

    assert fake_service.kwargs is not None
    assert fake_service.kwargs["auto_download"] is True
