# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ordered authored-image list loading orchestration."""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from simple_syrup.runtime.image_file_loader import ImageFileLoader
from simple_syrup.services.load_image_list_service import LoadImageListService


class FakeImageFileLoader(ImageFileLoader):
    """Provide deterministic images and fingerprints for service tests."""

    images: ClassVar[dict[str, torch.Tensor]] = {}
    loaded: ClassVar[list[str]] = []
    validated: ClassVar[list[str]] = []

    def available_files(self) -> tuple[str, ...]:
        """Return deterministic selectable paths."""

        return ("first.png", "second.png")

    def load(self, path: str) -> torch.Tensor:
        """Record and return one configured singleton image."""

        self.loaded.append(path)
        return self.images[path]

    def validate(self, path: str) -> None:
        """Record validation through the runtime boundary."""

        self.validated.append(path)

    def fingerprint(self, path: str) -> str:
        """Return a recognizable per-path fingerprint."""

        return f"digest:{path}"


class FakeLoadImageListService(LoadImageListService):
    """Use the fake runtime adapter in application-service tests."""

    loader_class = FakeImageFileLoader


def configured_service(images: dict[str, torch.Tensor]) -> FakeLoadImageListService:
    """Return a service configured with fake ordered images."""

    FakeImageFileLoader.images = images
    FakeImageFileLoader.loaded = []
    FakeImageFileLoader.validated = []
    return FakeLoadImageListService()


def test_loads_different_sized_images_as_ordered_singletons() -> None:
    """Image list members retain independent dimensions and authored order."""

    service = configured_service(
        {
            "wide.png": torch.full((1, 2, 4, 3), 0.8),
            "tall.png": torch.full((1, 5, 2, 3), 0.2),
        }
    )

    images = service.load(["wide.png", "tall.png"])

    assert isinstance(images, list)
    assert [tuple(image.shape) for image in images] == [(1, 2, 4, 3), (1, 5, 2, 3)]
    assert torch.all(images[0] == 0.8)
    assert torch.all(images[1] == 0.2)
    assert FakeImageFileLoader.loaded == ["wide.png", "tall.png"]


def test_preserves_duplicate_positions_and_rejects_empty_selection() -> None:
    """The list is positional rather than a set and must contain an image."""

    service = configured_service({"same.png": torch.ones((1, 2, 2, 3))})

    images = service.load(["same.png", "same.png"])

    assert len(images) == 2
    assert FakeImageFileLoader.loaded == ["same.png", "same.png"]
    with pytest.raises(ValueError, match="at least one image file"):
        service.load([])


def test_validation_and_fingerprint_are_order_sensitive() -> None:
    """Validation and caching preserve every selected position."""

    service = configured_service({})

    service.validate(["second.png", "first.png"])
    first = service.fingerprint(["first.png", "second.png"])
    reordered = service.fingerprint(["second.png", "first.png"])

    assert FakeImageFileLoader.validated == ["second.png", "first.png"]
    assert first != reordered
    assert service.available_files() == ("first.png", "second.png")
