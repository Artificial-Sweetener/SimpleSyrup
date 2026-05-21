# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for GroundingDINO text box detector adaptation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import torch

from simple_syrup.domain.segs import BoundingBox
from simple_syrup.runtime.grounding_dino_loader import GROUNDING_DINO_RUNTIME_PACKAGE
from simple_syrup.runtime.loaded_models import LoadedGroundingDINOModel
from simple_syrup.runtime.model_device_manager import TorchModelDeviceManager
from simple_syrup.runtime.text_box_detector import (
    GroundingDINOTextBoxDetector,
    TextBoxDetection,
)
from test_helpers import make_image_tensor


class PredictBoxesModel:
    """DINO-style object exposing a direct predict_boxes method."""

    def predict_boxes(
        self,
        image: torch.Tensor,
        prompt: str,
        threshold: float,
    ) -> torch.Tensor:
        """Return deterministic boxes."""

        return torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32)


def test_detector_accepts_predict_boxes_protocol() -> None:
    """Objects with predict_boxes are accepted with default confidence."""

    result = GroundingDINOTextBoxDetector().detect(
        PredictBoxesModel(),
        make_image_tensor(batch_size=1, height=2, width=2)[0],
        "face",
        0.3,
        "cpu",
    )

    assert result == (TextBoxDetection(bbox=BoundingBox(0, 0, 2, 2), confidence=1.0),)


def test_detector_accepts_loaded_grounding_dino_model(tmp_path: Path) -> None:
    """SimpleSyrup loaded DINO containers unwrap before adaptation."""

    loaded = LoadedGroundingDINOModel(
        model=PredictBoxesModel(),
        text_encoder_path=tmp_path / "bert",
        source="test",
        model_id="dino",
    )

    result = GroundingDINOTextBoxDetector().detect(
        loaded,
        make_image_tensor(batch_size=1, height=2, width=2)[0],
        "face",
        0.3,
        "cpu",
    )

    assert len(result) == 1
    assert result[0].confidence == 1.0


def test_detector_rejects_invalid_model() -> None:
    """Invalid DINO objects fail with an actionable error."""

    with pytest.raises(TypeError, match="Prompt SEGS w/ SAM"):
        GroundingDINOTextBoxDetector().detect(
            object(),
            make_image_tensor(batch_size=1, height=2, width=2)[0],
            "face",
            0.3,
            "cpu",
        )


def test_detector_rejects_invalid_box_shape() -> None:
    """Malformed predict_boxes output fails clearly."""

    class InvalidBoxesModel:
        """Return invalid boxes."""

        def predict_boxes(
            self,
            image: torch.Tensor,
            prompt: str,
            threshold: float,
        ) -> torch.Tensor:
            """Return malformed boxes."""

            return torch.ones((2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="invalid boxes shape"):
        GroundingDINOTextBoxDetector().detect(
            InvalidBoxesModel(),
            make_image_tensor(batch_size=1, height=2, width=2)[0],
            "face",
            0.3,
            "cpu",
        )


def test_detector_accepts_empty_boxes() -> None:
    """Empty predict_boxes output returns no detections."""

    class EmptyBoxesModel:
        """Return no boxes."""

        def predict_boxes(
            self,
            image: torch.Tensor,
            prompt: str,
            threshold: float,
        ) -> torch.Tensor:
            """Return an empty box tensor."""

            return torch.empty((0, 4), dtype=torch.float32)

    result = GroundingDINOTextBoxDetector().detect(
        EmptyBoxesModel(),
        make_image_tensor(batch_size=1, height=2, width=2)[0],
        "face",
        0.3,
        "cpu",
    )

    assert result == ()


def test_raw_grounding_dino_path_returns_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw GroundingDINO detections preserve max-logit confidence."""

    class FakeModel:
        """Raw callable GroundingDINO double."""

        def __call__(
            self,
            image: torch.Tensor,
            captions: list[str],
        ) -> dict[str, torch.Tensor]:
            """Return one high-confidence normalized box."""

            return {
                "pred_logits": torch.tensor([[[0.0, 4.0], [-4.0, -4.0]]]),
                "pred_boxes": torch.tensor(
                    [[[0.5, 0.5, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0]]]
                ),
            }

    class FakeTransform:
        """GroundingDINO transform double."""

        def __call__(
            self,
            image: object,
            target: object,
        ) -> tuple[torch.Tensor, object]:
            """Return a fake image tensor."""

            return torch.zeros((3, 2, 2), dtype=torch.float32), target

    transforms = ModuleType(f"{GROUNDING_DINO_RUNTIME_PACKAGE}.datasets.transforms")
    transforms.Compose = lambda steps: FakeTransform()  # type: ignore[attr-defined]
    transforms.RandomResize = lambda sizes, max_size: object()  # type: ignore[attr-defined]
    transforms.ToTensor = lambda: object()  # type: ignore[attr-defined]
    transforms.Normalize = lambda mean, std: object()  # type: ignore[attr-defined]
    comfy = ModuleType("comfy")
    model_management = ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: torch.device("cpu")  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        f"{GROUNDING_DINO_RUNTIME_PACKAGE}.datasets.transforms",
        transforms,
    )
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)

    result = GroundingDINOTextBoxDetector().detect(
        FakeModel(),
        make_image_tensor(batch_size=1, height=2, width=2)[0],
        "face",
        0.5,
        "cpu",
    )

    assert len(result) == 1
    assert result[0].bbox == BoundingBox(0, 0, 2, 2)
    assert result[0].confidence == pytest.approx(
        torch.sigmoid(torch.tensor(4.0)).item()
    )


def test_loaded_managed_grounding_dino_uses_comfy_device_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SimpleSyrup-loaded GroundingDINO moves model and image together for auto."""

    class FakeModel:
        """Raw callable GroundingDINO double with device recording."""

        def __init__(self) -> None:
            """Create a model fake."""

            self.device: torch.device | None = torch.device("cpu")
            self.image_devices: list[str] = []

        def eval(self) -> None:
            """Accept eval mode."""

        def __call__(
            self,
            image: torch.Tensor,
            captions: list[str],
        ) -> dict[str, torch.Tensor]:
            """Return no detections while recording the input device."""

            _ = captions
            self.image_devices.append(str(image.device))
            return {
                "pred_logits": torch.zeros((1, 1, 1), dtype=torch.float32),
                "pred_boxes": torch.zeros((1, 1, 4), dtype=torch.float32),
            }

    class FakeModelPatcher:
        """Fake Comfy patcher that places the model on the load device."""

        def __init__(
            self,
            model: object,
            load_device: torch.device,
            offload_device: torch.device,
        ) -> None:
            """Record patcher state."""

            self.model = model
            self.load_device = load_device
            self.offload_device = offload_device
            cast(Any, model).device = load_device

    class FakeTransform:
        """GroundingDINO transform double."""

        def __call__(
            self,
            image: object,
            target: object,
        ) -> tuple[torch.Tensor, object]:
            """Return a fake image tensor."""

            return torch.zeros((3, 2, 2), dtype=torch.float32), target

    state: dict[str, list[object]] = {"loaded": []}
    transforms = ModuleType(f"{GROUNDING_DINO_RUNTIME_PACKAGE}.datasets.transforms")
    transforms.Compose = lambda steps: FakeTransform()  # type: ignore[attr-defined]
    transforms.RandomResize = lambda sizes, max_size: object()  # type: ignore[attr-defined]
    transforms.ToTensor = lambda: object()  # type: ignore[attr-defined]
    transforms.Normalize = lambda mean, std: object()  # type: ignore[attr-defined]
    comfy = ModuleType("comfy")
    model_management = ModuleType("comfy.model_management")
    model_patcher = ModuleType("comfy.model_patcher")
    model_management.get_torch_device = lambda: torch.device("cpu")  # type: ignore[attr-defined]
    model_management.text_encoder_offload_device = lambda: torch.device("cpu")  # type: ignore[attr-defined]
    model_management.load_model_gpu = lambda patcher: state["loaded"].append(patcher)  # type: ignore[attr-defined]
    model_patcher.ModelPatcher = FakeModelPatcher  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        f"{GROUNDING_DINO_RUNTIME_PACKAGE}.datasets.transforms",
        transforms,
    )
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setitem(sys.modules, "comfy.model_patcher", model_patcher)

    model = FakeModel()
    managed = TorchModelDeviceManager().manage(model, "dino", "test")
    loaded = LoadedGroundingDINOModel(
        model=model,
        text_encoder_path=tmp_path / "bert",
        source="test",
        model_id="dino",
        managed_model=managed,
    )

    result = GroundingDINOTextBoxDetector().detect(
        loaded,
        make_image_tensor(batch_size=1, height=2, width=2)[0],
        "face",
        0.5,
        "auto",
    )

    assert result == ()
    assert model.image_devices == ["cpu"]
