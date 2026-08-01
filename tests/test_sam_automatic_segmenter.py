# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for unprompted SAM-family automatic segmentation runtime adapters."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from simple_syrup.runtime.sam_automatic_segmenter import SAMModelAutomaticSegmenter


def test_fast_sam_adapter_extracts_everything_masks_and_confidences() -> None:
    """FastSAM returns its direct mask results without text or CLIP prompting."""

    class _Masks:
        """Expose Ultralytics-style mask data."""

        data = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ]
        )

    class _Boxes:
        """Expose Ultralytics-style detection confidences."""

        conf = torch.tensor([0.75, 0.5])

    class _Result:
        """Expose one Ultralytics segmentation result."""

        masks = _Masks()
        boxes = _Boxes()

    class FastSAM:
        """Minimal FastSAM-like external model."""

        def __init__(self) -> None:
            """Create an empty prediction call recorder."""

            self.calls: list[dict[str, object]] = []

        def to(self, device: torch.device) -> None:
            """Accept temporary CPU inference movement."""

            del device

        def eval(self) -> None:
            """Accept evaluation mode."""

        def predict(self, image: object, **kwargs: object) -> list[_Result]:
            """Record direct everything-mode prediction options."""

            self.calls.append({"image": image, **kwargs})
            return [_Result()]

    model = FastSAM()

    masks = SAMModelAutomaticSegmenter().segment_all(
        model,
        torch.zeros((1, 2, 2, 3)),
        execution_device="cpu",
    )

    assert [mask.confidence for mask in masks] == [0.75, 0.5]
    assert torch.equal(masks[0].mask, _Masks.data[0])
    assert model.calls[0]["retina_masks"] is True
    assert model.calls[0]["verbose"] is False
    assert model.calls[0]["device"] == "cpu"


def test_segment_anything_adapter_preserves_predicted_iou(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standard SAM automatic generation maps quality metadata to SEGS confidence."""

    class _Generator:
        """Return a fixed native automatic-mask response."""

        def __init__(self, model: object) -> None:
            """Accept the raw SAM model."""

            del model

        def generate(self, image: object) -> list[dict[str, object]]:
            """Return one native SAM automatic mask."""

            del image
            return [
                {
                    "segmentation": torch.tensor([[True, False], [False, True]]),
                    "predicted_iou": 0.92,
                }
            ]

    segment_anything = ModuleType("segment_anything")
    segment_anything.SamAutomaticMaskGenerator = _Generator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "segment_anything", segment_anything)

    class _SAM:
        """Minimal raw SAM model."""

        model_name = "sam_vit_b.pth"

        def to(self, device: torch.device) -> None:
            """Accept temporary CPU inference movement."""

            del device

        def eval(self) -> None:
            """Accept evaluation mode."""

    masks = SAMModelAutomaticSegmenter().segment_all(
        _SAM(),
        torch.zeros((1, 2, 2, 3)),
        execution_device="cpu",
    )

    assert len(masks) == 1
    assert masks[0].confidence == 0.92
    assert torch.equal(
        masks[0].mask,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
