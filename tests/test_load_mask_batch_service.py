# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ordered authored-mask loading orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import pytest
import torch

from simple_syrup.runtime.mask_file_loader import MaskFileLoader
from simple_syrup.services.load_mask_batch_service import LoadMaskBatchService


class FakeMaskFileLoader(MaskFileLoader):
    """Provide deterministic masks and fingerprints for service tests."""

    masks: ClassVar[dict[str, torch.Tensor]] = {}
    loaded: ClassVar[list[tuple[str, str]]] = []
    validated: ClassVar[list[tuple[str, str]]] = []

    def available_files(self) -> tuple[str, ...]:
        """Return deterministic selectable paths."""

        return ("first.png", "second.png")

    def load(self, path: str, channel: str) -> torch.Tensor:
        """Record and return the configured singleton mask."""

        self.loaded.append((path, channel))
        return self.masks[path]

    def validate(self, path: str, channel: str) -> None:
        """Record validation through the runtime boundary."""

        self.validated.append((path, channel))

    def fingerprint(self, path: str) -> str:
        """Return a recognizable per-path fingerprint."""

        return f"digest:{path}"


class FakeLoadMaskBatchService(LoadMaskBatchService):
    """Use the fake runtime adapter in application-service tests."""

    loader_class = FakeMaskFileLoader


def _service_with_masks(masks: dict[str, torch.Tensor]) -> FakeLoadMaskBatchService:
    """Return a service configured with fake ordered masks."""

    FakeMaskFileLoader.masks = masks
    FakeMaskFileLoader.loaded = []
    FakeMaskFileLoader.validated = []
    return FakeLoadMaskBatchService()


def test_loads_one_or_many_files_in_supplied_order() -> None:
    """The file sequence directly defines MASK batch order."""

    service = _service_with_masks(
        {
            "b.png": torch.full((1, 2, 3), 0.8),
            "a.png": torch.full((1, 2, 3), 0.2),
        }
    )

    one = service.load(["a.png"], "red")
    many = service.load(["b.png", "a.png"], "blue")

    assert one.shape == (1, 2, 3)
    assert torch.all(one == 0.2)
    assert torch.all(many[0] == 0.8)
    assert torch.all(many[1] == 0.2)
    assert FakeMaskFileLoader.loaded == [
        ("a.png", "red"),
        ("b.png", "blue"),
        ("a.png", "blue"),
    ]


def test_rejects_empty_selection_and_dimension_mismatch() -> None:
    """A batch always contains compatible one-file regions."""

    service = _service_with_masks(
        {
            "small.png": torch.zeros((1, 2, 2)),
            "wide.png": torch.zeros((1, 2, 3)),
        }
    )

    with pytest.raises(ValueError, match="at least one mask file"):
        service.load([], "alpha")
    with pytest.raises(ValueError, match="identical dimensions"):
        service.load(["small.png", "wide.png"], "alpha")


def test_fingerprint_is_sensitive_to_order_channel_and_contents() -> None:
    """Caching distinguishes every workflow-relevant loader input."""

    service = _service_with_masks({})

    first = service.fingerprint(["a.png", "b.png"], "red")
    reordered = service.fingerprint(["b.png", "a.png"], "red")
    other_channel = service.fingerprint(["a.png", "b.png"], "blue")

    assert first != reordered
    assert first != other_channel


def test_available_files_delegates_to_runtime_adapter() -> None:
    """Schema choices come from the Comfy filesystem boundary."""

    assert FakeLoadMaskBatchService().available_files() == (
        "first.png",
        "second.png",
    )


def test_validation_preserves_selection_order() -> None:
    """Every native widget value is validated in its authored order."""

    service = _service_with_masks({})

    service.validate(["second.png", "first.png"], "green")

    assert FakeMaskFileLoader.validated == [
        ("second.png", "green"),
        ("first.png", "green"),
    ]


@pytest.mark.parametrize("files", ["single.png", ("single.png",)])
def test_string_and_sequence_inputs_both_mean_one_mask(
    files: str | Sequence[str],
) -> None:
    """API callers may serialize one selected path as a scalar or list."""

    service = _service_with_masks({"single.png": torch.ones((1, 2, 2))})

    result = service.load(files, "alpha")

    assert result.shape == (1, 2, 2)
