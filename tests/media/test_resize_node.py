# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Resize Image to Target ComfyUI node declaration."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.nodes.image_resize_to_target import ResizeImageToTarget


def test_resize_node_contract_constants() -> None:
    """Node constants match the public ComfyUI contract."""

    assert ResizeImageToTarget.RETURN_TYPES == ("IMAGE", "INT", "INT", "MASK")
    assert ResizeImageToTarget.RETURN_NAMES == ("image", "width", "height", "mask")
    assert ResizeImageToTarget.FUNCTION == "resize"
    assert ResizeImageToTarget.CATEGORY == "SimpleSyrup/Image"


def test_resize_node_declares_expected_inputs() -> None:
    """Node input declaration includes required controls and optional mask."""

    input_types: dict[str, dict[str, tuple[Any, ...]]] = (
        ResizeImageToTarget.INPUT_TYPES()
    )

    required = input_types["required"]
    optional = input_types["optional"]

    assert set(required) == {
        "image",
        "width",
        "height",
        "resize_mode",
        "sampling",
        "processor",
        "divisible_by",
        "crop_position",
        "pad_color",
        "max_batch_size",
        "sinc_window",
        "precision",
    }
    assert set(optional) == {"mask"}
    assert required["resize_mode"][0] == [
        "Stretch",
        "Keep AR",
        "Crop (Cover + Crop)",
        "Pad (Fit + Pad)",
    ]
    assert required["sampling"][0] == [
        "nearest-exact",
        "bilinear",
        "area",
        "bicubic",
        "lanczos",
    ]
    assert required["processor"][0] == ["cpu", "gpu"]


def test_resize_node_delegates_to_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Node execution delegates directly to the resize service."""

    calls: list[dict[str, object]] = []
    expected = (
        torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        2,
        2,
        torch.zeros((1, 2, 2), dtype=torch.float32),
    )

    class FakeService:
        """Service double used to verify node delegation."""

        def resize(
            self, **kwargs: object
        ) -> tuple[torch.Tensor, int, int, torch.Tensor]:
            """Record delegated keyword arguments."""

            calls.append(kwargs)
            return expected

    monkeypatch.setattr(ResizeImageToTarget, "_service", FakeService())
    image = torch.ones((1, 4, 4, 3), dtype=torch.float32)
    mask = torch.ones((1, 4, 4), dtype=torch.float32)

    result = ResizeImageToTarget().resize(
        image=image,
        width=2,
        height=2,
        resize_mode="Stretch",
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

    assert result == expected
    assert calls == [
        {
            "image": image,
            "width": 2,
            "height": 2,
            "resize_mode": "Stretch",
            "sampling": "nearest-exact",
            "processor": "cpu",
            "divisible_by": 1,
            "crop_position": "center",
            "pad_color": "0, 0, 0",
            "max_batch_size": 0,
            "sinc_window": 3,
            "precision": "fp32",
            "mask": mask,
        }
    ]
