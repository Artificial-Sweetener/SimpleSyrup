# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ViTMatte mask refinement."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from simple_syrup.masking.mask_ops import MaskRefinementSettings
from simple_syrup.runtime.loaded_models import LoadedViTMatteModel
from simple_syrup.runtime.model_device_manager import TorchModelDeviceManager
from simple_syrup.runtime.vitmatte_refiner import (
    ViTMatteRefiner,
    generate_vitmatte_trimap,
)
from test_helpers import make_image_tensor


class FakeProcessor:
    """Processor double that returns tensor inputs."""

    def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
        """Return deterministic tensor inputs."""

        return {"pixel_values": torch.zeros((1, 3, 2, 2), dtype=torch.float32)}


class FakeModel:
    """ViTMatte model double."""

    def __init__(self) -> None:
        """Create a recording fake model."""

        self.devices: list[str] = []

    def to(self, device: object) -> None:
        """Record device movement."""

        self.devices.append(str(device))

    def eval(self) -> None:
        """Accept eval mode."""

    def __call__(self, **kwargs: object) -> object:
        """Return a deterministic alpha matte."""

        return SimpleNamespace(alphas=torch.ones((1, 1, 2, 2), dtype=torch.float32))


def test_generate_vitmatte_trimap_marks_unknown_band() -> None:
    """Trimap generation creates background, foreground, and unknown values."""

    mask = torch.zeros((7, 7), dtype=torch.float32)
    mask[2:5, 2:5] = 1.0

    trimap = generate_vitmatte_trimap(mask, erode_radius=1, dilate_radius=1)

    assert 0.0 in trimap
    assert 0.5 in trimap
    assert 1.0 in trimap


def test_vitmatte_refiner_requires_connected_model() -> None:
    """VITMatte refinement fails clearly without a VITMATTE_MODEL."""

    with pytest.raises(ValueError, match="requires a connected VITMATTE_MODEL"):
        ViTMatteRefiner().refine(
            make_image_tensor(batch_size=1, height=2, width=2),
            torch.ones((1, 2, 2), dtype=torch.float32),
            _settings(),
            None,
        )


def test_vitmatte_refiner_runs_model_and_preserves_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ViTMatte refinement returns a BHW mask at the original size."""

    _install_fake_comfy_device(monkeypatch)
    model = FakeModel()
    loaded = LoadedViTMatteModel(
        model=model,
        processor=FakeProcessor(),
        source="test",
        model_id="vitmatte-small-composition-1k",
        model_path=tmp_path,
        managed_model=TorchModelDeviceManager().manage(
            model,
            "vitmatte-small-composition-1k",
            "test",
        ),
    )

    result = ViTMatteRefiner().refine(
        make_image_tensor(batch_size=1, height=4, width=4),
        torch.ones((1, 4, 4), dtype=torch.float32),
        _settings(max_size_pixels=4),
        loaded,
    )

    assert result.shape == (1, 4, 4)
    assert torch.all(result == 1.0)
    assert "cpu" in model.devices


def _settings(max_size_pixels: int = 16) -> MaskRefinementSettings:
    """Return common ViTMatte refinement settings."""

    return MaskRefinementSettings(
        detail_method="VITMatte",
        detail_erode=1,
        detail_dilate=1,
        black_point=0.0,
        white_point=1.0,
        process_detail=True,
        execution_device="cpu",
        max_size_pixels=max_size_pixels,
    )


def _install_fake_comfy_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install fake Comfy model management for auto device paths."""

    comfy = ModuleType("comfy")
    model_management = ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: torch.device("cpu")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
