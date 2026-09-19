# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify decoded NegPiP proof artifacts and contact-sheet evidence."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from tools.comfy_api import JsonObject
from tools.negpip_integration.visual_proof import NegpipVisualProofRecorder
from tools.negpip_integration.workflow import NegpipLiveFamily


def test_visual_recorder_persists_pairs_and_contact_sheet(tmp_path: Path) -> None:
    """Every family receives originals, pixel deltas, and visible labels."""

    recorder = NegpipVisualProofRecorder(tmp_path)
    families: JsonObject = {}
    for index, family in enumerate(NegpipLiveFamily):
        control = recorder.record(
            family,
            "control",
            _png_bytes((20 + index, 40, 60)),
        )
        negative = recorder.record(
            family,
            "negative",
            _png_bytes((120 + index, 40, 60)),
        )
        families[family.value] = {
            "control": {"image": control},
            "negative": {
                "image": negative,
                "runtime": {
                    "family": family.value,
                    "attention_calls": 3,
                    "negative_mask_calls": 3,
                },
            },
        }

    sheet = recorder.finalize(families)

    assert (tmp_path / str(sheet["file"])).is_file()
    assert sheet["width"] == 1346
    assert sheet["height"] == 2528
    for family in NegpipLiveFamily:
        result = families[family.value]
        assert isinstance(result, dict)
        comparison = result["image_comparison"]
        assert isinstance(comparison, dict)
        assert comparison["changed_pixels"] == 512 * 512
        assert comparison["mean_absolute_rgb_delta"] > 0


def test_visual_recorder_rejects_identical_pair(tmp_path: Path) -> None:
    """A decoded image must visibly change in every supported family."""

    recorder = NegpipVisualProofRecorder(tmp_path)
    image = _png_bytes((20, 40, 60))
    families: JsonObject = {}
    for family in NegpipLiveFamily:
        families[family.value] = {
            "control": {"image": recorder.record(family, "control", image)},
            "negative": {
                "image": recorder.record(family, "negative", image),
                "runtime": {
                    "family": family.value,
                    "attention_calls": 1,
                    "negative_mask_calls": 1,
                },
            },
        }

    try:
        recorder.finalize(families)
    except ValueError as error:
        assert "images are equal" in str(error)
    else:
        raise AssertionError("Identical NegPiP proof images must be rejected.")


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    """Return one deterministic 512-square PNG."""

    stream = io.BytesIO()
    Image.new("RGB", (512, 512), color).save(stream, format="PNG")
    return stream.getvalue()
