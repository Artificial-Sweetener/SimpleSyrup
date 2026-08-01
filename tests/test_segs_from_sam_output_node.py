# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the SEGS from SAM Output ComfyUI node."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.nodes.segs_from_sam_output import SEGSFromSAMOutput


def test_node_declares_automatic_sam_to_segs_contract() -> None:
    """The node exposes standard IMAGE, SAM_MODEL, and SEGS sockets."""

    inputs = SEGSFromSAMOutput.INPUT_TYPES()["required"]

    assert tuple(inputs) == (
        "image",
        "sam_model",
        "segmentation_resolution",
        "minimum_region_area",
    )
    assert inputs["segmentation_resolution"][1]["default"] == 640
    assert inputs["segmentation_resolution"][1]["step"] == 64
    assert SEGSFromSAMOutput.RETURN_TYPES == ("SEGS",)
    assert SEGSFromSAMOutput.OUTPUT_IS_LIST == (True,)


def test_node_builds_one_segs_output_per_image_batch_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic segmentation remains aligned with ComfyUI IMAGE batches."""

    calls: list[torch.Tensor] = []
    phase_progress = _RecordingPhaseProgress()

    class _Service:
        """Record image batches and return a distinct opaque SEGS payload."""

        def build(self, **kwargs: object) -> object:
            """Record one source image and return it as an opaque marker."""

            image = kwargs["image"]
            assert isinstance(image, torch.Tensor)
            calls.append(image)
            progress = kwargs["phase_progress"]
            assert progress is phase_progress
            for phase in (
                "preparing_segmentation_image",
                "generating_automatic_masks",
                "building_segs",
            ):
                phase_progress.advance(phase)
            return (image.shape[1:3], ())

    monkeypatch.setattr(SEGSFromSAMOutput, "service_class", _Service)
    monkeypatch.setattr(
        SEGSFromSAMOutput,
        "progress_factory",
        lambda **_kwargs: phase_progress,
    )

    (segs,) = SEGSFromSAMOutput().generate(
        image=torch.zeros((2, 16, 16, 3)),
        sam_model=object(),
        segmentation_resolution=640,
        minimum_region_area=0,
    )

    assert len(calls) == 2
    assert len(segs) == 2
    assert phase_progress.phases == [
        "preparing_segmentation_image",
        "generating_automatic_masks",
        "building_segs",
        "preparing_segmentation_image",
        "generating_automatic_masks",
        "building_segs",
        "completed",
    ]


class _RecordingPhaseProgress:
    """Record node progress without constructing a real ComfyUI progress bar."""

    def __init__(self) -> None:
        """Create an empty phase record."""

        self.phases: list[str] = []

    def advance(self, phase: str) -> None:
        """Record one phase transition."""

        self.phases.append(phase)
