# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for SAM model compatibility adaptation."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from simple_syrup.runtime.loaded_models import LoadedSAMModel
from simple_syrup.runtime.model_device_manager import TorchModelDeviceManager
from simple_syrup.runtime.sam_segmenter import SAMModelSegmenter
from test_helpers import make_image_tensor


class RecordingWrapper:
    """Impact-style SAM wrapper double."""

    def __init__(self) -> None:
        """Create a recording wrapper."""

        self.prepared = False
        self.released = False
        self.boxes: list[list[float]] = []

    def prepare_device(self) -> None:
        """Record preparation."""

        self.prepared = True

    def release_device(self) -> None:
        """Record release."""

        self.released = True

    def predict(
        self,
        image: object,
        points: list[object],
        plabs: list[int],
        bbox: list[float],
        threshold: float,
    ) -> list[torch.Tensor]:
        """Return one deterministic mask."""

        self.boxes.append(bbox)
        return [torch.ones((2, 2), dtype=torch.float32)]


def test_segmenter_accepts_impact_style_sam_wrapper() -> None:
    """Objects with `.sam_wrapper` are accepted without importing Impact Pack."""

    wrapper = RecordingWrapper()
    model = SimpleNamespace(sam_wrapper=wrapper)

    result = SAMModelSegmenter().segment_boxes(
        model,
        make_image_tensor(batch_size=1, height=2, width=2)[0],
        torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32),
        threshold=0.3,
        execution_device="cpu",
    )

    assert wrapper.prepared is True
    assert wrapper.released is True
    assert len(result) == 1
    assert torch.equal(result[0], torch.ones((2, 2), dtype=torch.float32))


def test_segmenter_accepts_loaded_sam_model_wrapper() -> None:
    """SimpleSyrup loaded SAM containers unwrap before adaptation."""

    wrapper = RecordingWrapper()
    loaded = LoadedSAMModel(
        model=SimpleNamespace(sam_wrapper=wrapper),
        source="test",
        model_id="sam",
    )

    result = SAMModelSegmenter().segment_boxes(
        loaded,
        make_image_tensor(batch_size=1, height=2, width=2)[0],
        torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32),
        threshold=0.3,
        execution_device="cpu",
    )

    assert len(result) == 1
    assert result[0].shape == (2, 2)


def test_segmenter_returns_empty_tuple_for_empty_boxes() -> None:
    """Empty boxes do not require a SAM-compatible object."""

    result = SAMModelSegmenter().segment_boxes(
        object(),
        make_image_tensor(batch_size=1, height=2, width=3)[0],
        torch.empty((0, 4), dtype=torch.float32),
        threshold=0.3,
        execution_device="cpu",
    )

    assert result == ()


def test_segmenter_preserves_one_mask_per_box() -> None:
    """Wrapper predictions produce one mask for each requested box."""

    wrapper = RecordingWrapper()

    result = SAMModelSegmenter().segment_boxes(
        SimpleNamespace(sam_wrapper=wrapper),
        make_image_tensor(batch_size=1, height=2, width=2)[0],
        torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]],
            dtype=torch.float32,
        ),
        threshold=0.3,
        execution_device="cpu",
    )

    assert len(result) == 2
    assert wrapper.boxes == [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]]


def test_segmenter_combines_candidate_masks_per_box() -> None:
    """Multiple masks from one wrapper call are unioned for that box only."""

    class CandidateWrapper(RecordingWrapper):
        """Return two candidates per requested box."""

        def predict(
            self,
            image: object,
            points: list[object],
            plabs: list[int],
            bbox: list[float],
            threshold: float,
        ) -> list[torch.Tensor]:
            """Return two complementary masks."""

            return [
                torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
                torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
            ]

    result = SAMModelSegmenter().segment_boxes(
        SimpleNamespace(sam_wrapper=CandidateWrapper()),
        make_image_tensor(batch_size=1, height=2, width=2)[0],
        torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32),
        threshold=0.3,
        execution_device="cpu",
    )

    assert torch.equal(result[0], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_segmenter_accepts_raw_segment_anything_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw SAM models use the segment_anything predictor path."""

    class FakePredictor:
        """Small SamPredictor fake."""

        def __init__(self, model: object) -> None:
            """Create fake predictor."""

            self.transform = SimpleNamespace(
                apply_boxes_torch=lambda boxes, shape: boxes
            )

        def set_image(self, image: object) -> None:
            """Accept an image."""

        def predict_torch(self, **kwargs: object) -> tuple[torch.Tensor, None, None]:
            """Return a deterministic mask batch."""

            return (
                torch.tensor(
                    [
                        [[[1.0, 0.0], [0.0, 0.0]]],
                        [[[0.0, 0.0], [0.0, 1.0]]],
                    ],
                    dtype=torch.float32,
                ),
                None,
                None,
            )

    segment_anything = ModuleType("segment_anything")
    segment_anything.SamPredictor = FakePredictor  # type: ignore[attr-defined]
    comfy = ModuleType("comfy")
    model_management = ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: torch.device("cpu")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "segment_anything", segment_anything)
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)

    result = SAMModelSegmenter().segment_boxes(
        object(),
        make_image_tensor(batch_size=1, height=2, width=2)[0],
        torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]],
            dtype=torch.float32,
        ),
        threshold=0.3,
        execution_device="cpu",
    )

    assert len(result) == 2
    assert torch.equal(result[0], torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
    assert torch.equal(result[1], torch.tensor([[0.0, 0.0], [0.0, 1.0]]))


def test_segmenter_uses_manager_for_loaded_raw_sam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SimpleSyrup-loaded raw SAM models use the shared device context."""

    class FakeModel:
        """SAM model fake that records movement."""

        def __init__(self) -> None:
            """Create the fake."""

            self.to_calls: list[str] = []

        def to(self, device: object) -> None:
            """Record device movement."""

            self.to_calls.append(str(device))

        def eval(self) -> None:
            """Accept eval mode."""

    class FakePredictor:
        """Small SamPredictor fake."""

        def __init__(self, model: object) -> None:
            """Create fake predictor."""

            self.transform = SimpleNamespace(
                apply_boxes_torch=lambda boxes, shape: boxes
            )

        def set_image(self, image: object) -> None:
            """Accept an image."""

        def predict_torch(self, **kwargs: object) -> tuple[torch.Tensor, None, None]:
            """Return a deterministic mask batch."""

            return (torch.ones((1, 1, 2, 2), dtype=torch.float32), None, None)

    segment_anything = ModuleType("segment_anything")
    segment_anything.SamPredictor = FakePredictor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "segment_anything", segment_anything)

    model = FakeModel()
    loaded = LoadedSAMModel(
        model=model,
        source="test",
        model_id="sam",
        managed_model=TorchModelDeviceManager().manage(model, "sam", "test"),
    )

    result = SAMModelSegmenter().segment_boxes(
        loaded,
        make_image_tensor(batch_size=1, height=2, width=2)[0],
        torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
        threshold=0.3,
        execution_device="cpu",
    )

    assert len(result) == 1
    assert model.to_calls == ["cpu"]


def test_segmenter_uses_hq_predictor_for_loaded_sam_hq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SimpleSyrup-loaded SAM-HQ models use the vendored HQ predictor."""

    class FakeModel:
        """SAM-HQ model fake that records movement."""

        def __init__(self) -> None:
            """Create the fake model."""

            self.to_calls: list[str] = []

        def to(self, device: object) -> None:
            """Record device movement."""

            self.to_calls.append(str(device))

        def eval(self) -> None:
            """Accept eval mode."""

    class FakeHQPredictor:
        """Small SamPredictorHQ fake."""

        used_hq_mode = False

        def __init__(self, model: object, sam_is_hq: bool = False) -> None:
            """Record whether the HQ flag was requested."""

            self.transform = SimpleNamespace(
                apply_boxes_torch=lambda boxes, shape: boxes
            )
            FakeHQPredictor.used_hq_mode = sam_is_hq

        def set_image(self, image: object) -> None:
            """Accept an image."""

        def predict_torch(self, **kwargs: object) -> tuple[torch.Tensor, None, None]:
            """Return a deterministic mask batch."""

            return (torch.ones((1, 1, 2, 2), dtype=torch.float32), None, None)

    predictor_module = ModuleType("simple_syrup.third_party.sam_hq_runtime.predictor")
    predictor_module.SamPredictorHQ = FakeHQPredictor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, predictor_module.__name__, predictor_module)

    model = FakeModel()
    loaded = LoadedSAMModel(
        model=model,
        source="test",
        model_id="sam_hq_vit_b",
        managed_model=TorchModelDeviceManager().manage(
            model,
            "sam_hq_vit_b",
            "test",
        ),
    )

    result = SAMModelSegmenter().segment_boxes(
        loaded,
        make_image_tensor(batch_size=1, height=2, width=2)[0],
        torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
        threshold=0.3,
        execution_device="cpu",
    )

    assert len(result) == 1
    assert FakeHQPredictor.used_hq_mode is True
    assert model.to_calls == ["cpu"]


def test_segmenter_rejects_invalid_mask_shape() -> None:
    """Invalid wrapper mask shapes fail clearly."""

    class InvalidWrapper(RecordingWrapper):
        """Return an invalid mask shape."""

        def predict(
            self,
            image: object,
            points: list[object],
            plabs: list[int],
            bbox: list[float],
            threshold: float,
        ) -> list[torch.Tensor]:
            """Return a one-dimensional mask."""

            return [torch.ones((2,), dtype=torch.float32)]

    with pytest.raises(ValueError, match="invalid mask shape"):
        SAMModelSegmenter().segment_boxes(
            SimpleNamespace(sam_wrapper=InvalidWrapper()),
            make_image_tensor(batch_size=1, height=2, width=2)[0],
            torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32),
            threshold=0.3,
            execution_device="cpu",
        )
