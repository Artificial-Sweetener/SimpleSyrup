# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the resize image application service."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from simple_syrup.image.resize_service import (
    ResizeImageToTargetService,
    parse_pad_color,
)


@dataclass
class RecordingProgress:
    """Test progress reporter that records update calls."""

    total: int
    updates: list[int]

    def update(self, value: int) -> None:
        """Record a progress update."""

        self.updates.append(value)


def test_service_resizes_image_and_returns_shape_metadata() -> None:
    """Service returns BHWC image output plus matching width and height."""

    service = _service_with_progress([])
    image = torch.ones((2, 4, 6, 3), dtype=torch.float32)

    output, width, height, mask = service.resize(
        image=image,
        width=8,
        height=10,
        resize_mode="Stretch",
        sampling="nearest-exact",
        processor="cpu",
        divisible_by=1,
        crop_position="center",
        pad_color="0, 0, 0",
        max_batch_size=0,
        sinc_window=3,
        precision="fp32",
    )

    assert output.shape == (2, 10, 8, 3)
    assert (width, height) == (8, 10)
    assert mask.shape == (2, 10, 8)
    assert torch.count_nonzero(mask) == 0


def test_service_resizes_mask_with_same_geometry() -> None:
    """Provided masks are resized and returned with output geometry."""

    service = _service_with_progress([])
    image = torch.ones((1, 4, 6, 3), dtype=torch.float32)
    mask = torch.ones((1, 4, 6), dtype=torch.float32)

    _output, width, height, resized_mask = service.resize(
        image=image,
        width=8,
        height=8,
        resize_mode="Pad (Fit + Pad)",
        sampling="nearest-exact",
        processor="cpu",
        divisible_by=1,
        crop_position="center",
        pad_color="0, 0, 0",
        max_batch_size=0,
        sinc_window=3,
        precision="fp32",
        mask=mask,
    )

    assert (width, height) == (8, 8)
    assert resized_mask.shape == (1, 8, 8)
    assert torch.all(resized_mask[:, 1:6, :] == 1.0)
    assert torch.all(resized_mask[:, 0, :] == 0.0)
    assert torch.all(resized_mask[:, 6:, :] == 0.0)


def test_service_uses_pad_color_for_image_padding() -> None:
    """Pad mode fills image padding with parsed RGB values."""

    service = _service_with_progress([])
    image = torch.zeros((1, 2, 4, 3), dtype=torch.float32)

    output, _width, _height, _mask = service.resize(
        image=image,
        width=4,
        height=4,
        resize_mode="Pad (Fit + Pad)",
        sampling="nearest-exact",
        processor="cpu",
        divisible_by=1,
        crop_position="center",
        pad_color="255, 0, 128",
        max_batch_size=0,
        sinc_window=3,
        precision="fp32",
    )

    expected = torch.tensor([1.0, 0.0, 128.0 / 255.0], dtype=torch.float32)
    assert torch.allclose(output[0, 0, 0], expected)
    assert torch.allclose(output[0, -1, 0], expected)


def test_service_chunking_preserves_batch_order() -> None:
    """Chunked execution concatenates output chunks in input order."""

    progress_updates: list[int] = []
    service = _service_with_progress(progress_updates)
    image = torch.stack(
        [
            torch.full((2, 2, 3), 0.1, dtype=torch.float32),
            torch.full((2, 2, 3), 0.5, dtype=torch.float32),
            torch.full((2, 2, 3), 0.9, dtype=torch.float32),
        ],
        dim=0,
    )

    output, _width, _height, _mask = service.resize(
        image=image,
        width=2,
        height=2,
        resize_mode="Stretch",
        sampling="nearest-exact",
        processor="cpu",
        divisible_by=1,
        crop_position="center",
        pad_color="0, 0, 0",
        max_batch_size=1,
        sinc_window=3,
        precision="fp32",
    )

    assert torch.allclose(output[:, 0, 0, 0], torch.tensor([0.1, 0.5, 0.9]))
    assert progress_updates == [1, 1, 1]


def test_parse_pad_color_supports_gray_rgb_and_rgba() -> None:
    """Pad color parsing adapts to supported channel counts."""

    assert torch.allclose(parse_pad_color("255, 128, 0", 1), torch.tensor([1.0]))
    assert torch.allclose(
        parse_pad_color("255, 128, 0", 3),
        torch.tensor([1.0, 128.0 / 255.0, 0.0]),
    )
    assert torch.allclose(
        parse_pad_color("255, 128, 0", 4),
        torch.tensor([1.0, 128.0 / 255.0, 0.0, 1.0]),
    )


def test_parse_pad_color_rejects_malformed_values() -> None:
    """Malformed pad color strings fail before image processing."""

    with pytest.raises(ValueError, match="exactly three"):
        parse_pad_color("0, 0", 3)
    with pytest.raises(ValueError, match="not an integer"):
        parse_pad_color("0, bad, 0", 3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"image": torch.ones((1, 4, 6), dtype=torch.float32)}, "shape"),
        ({"image": torch.ones((1, 4, 6, 2), dtype=torch.float32)}, "channel"),
        ({"processor": "bad"}, "processor"),
        ({"sampling": "bad"}, "sampling"),
        ({"mask": torch.ones((2, 4, 6), dtype=torch.float32)}, "batch size"),
    ],
)
def test_service_rejects_invalid_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Invalid service inputs produce clear errors."""

    service = _service_with_progress([])
    base_kwargs: dict[str, object] = {
        "image": torch.ones((1, 4, 6, 3), dtype=torch.float32),
        "width": 8,
        "height": 10,
        "resize_mode": "Stretch",
        "sampling": "nearest-exact",
        "processor": "cpu",
        "divisible_by": 1,
        "crop_position": "center",
        "pad_color": "0, 0, 0",
        "max_batch_size": 0,
        "sinc_window": 3,
        "precision": "fp32",
        "mask": None,
    }
    base_kwargs.update(kwargs)

    with pytest.raises((TypeError, ValueError), match=match):
        service.resize(**base_kwargs)  # type: ignore[arg-type]


def _service_with_progress(progress_updates: list[int]) -> ResizeImageToTargetService:
    """Create a service that uses test progress collection."""

    def progress_factory(total: int) -> RecordingProgress:
        """Return a recording progress reporter."""

        return RecordingProgress(total=total, updates=progress_updates)

    return ResizeImageToTargetService(progress_factory=progress_factory)
