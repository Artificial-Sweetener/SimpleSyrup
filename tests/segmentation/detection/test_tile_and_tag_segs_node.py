# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Tile & Tag SEGS node contract."""

from __future__ import annotations

from typing import Any, cast

import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment
from simple_syrup.domain.tile_segs import TileSEGSControls
from simple_syrup.nodes.tile_and_tag_segs import (
    DEFAULT_EXCLUDE_TAGS,
    TileAndTagSEGS,
)
from simple_syrup.runtime.wd14_tagger import WD14TagFormattingControls
from simple_syrup.services.tile_and_tag_segs_service import TileAndTagSEGSResult


def test_tile_and_tag_segs_contract() -> None:
    """Tile & Tag SEGS exposes the agreed ComfyUI contract."""

    inputs = TileAndTagSEGS.INPUT_TYPES()

    assert TileAndTagSEGS.RETURN_TYPES == ("SEGS", "CONDITIONING_BATCH")
    assert TileAndTagSEGS.RETURN_NAMES == ("segs", "positive")
    assert TileAndTagSEGS.FUNCTION == "tile_and_tag"
    assert TileAndTagSEGS.CATEGORY == "SimpleSyrup/Detailing"
    assert list(inputs["required"]) == [
        "image",
        "clip",
        "wd14_tagger",
        "universal_positive",
        "bbox_size",
        "crop_factor",
        "min_overlap",
        "filter_segs_dilation",
        "mask_irregularity",
        "irregular_mask_mode",
        "threshold",
        "character_threshold",
        "replace_underscore",
        "trailing_comma",
        "exclude_tags",
    ]
    assert "optional" not in inputs
    assert inputs["required"]["wd14_tagger"][0] == "WD14_TAGGER"
    assert inputs["required"]["universal_positive"][0] == "STRING"
    assert inputs["required"]["universal_positive"][1]["default"] == ""
    assert inputs["required"]["universal_positive"][1]["multiline"] is False
    assert inputs["required"]["bbox_size"][1]["default"] == 872
    assert inputs["required"]["crop_factor"][1]["default"] == 1.1
    assert inputs["required"]["min_overlap"][1]["default"] == 16
    assert inputs["required"]["filter_segs_dilation"][1]["default"] == 20
    assert inputs["required"]["mask_irregularity"][1]["default"] == 0
    assert inputs["required"]["irregular_mask_mode"][0][0] == "Reuse fast"
    assert inputs["required"]["threshold"][1]["default"] == 0.35
    assert inputs["required"]["character_threshold"][1]["default"] == 1.0
    assert inputs["required"]["replace_underscore"][1]["default"] is True
    assert inputs["required"]["trailing_comma"][1]["default"] is False
    assert inputs["required"]["exclude_tags"][1]["default"] == DEFAULT_EXCLUDE_TAGS
    assert inputs["required"]["exclude_tags"][1]["multiline"] is False


def test_tile_and_tag_segs_delegates_to_service(
    monkeypatch: Any,
) -> None:
    """The node delegates behavior and returns service outputs unchanged."""

    service = _FakeService()
    monkeypatch.setattr(TileAndTagSEGS, "service_class", lambda: service)
    image = torch.zeros((1, 8, 8, 3))
    wd14_tagger = object()

    segs, positive = TileAndTagSEGS().tile_and_tag(
        image=image,
        clip="clip",
        wd14_tagger=wd14_tagger,
        universal_positive="masterpiece",
        bbox_size=872,
        crop_factor=1.1,
        min_overlap=16,
        filter_segs_dilation=20,
        mask_irregularity=0.0,
        irregular_mask_mode="Reuse fast",
        threshold=0.35,
        character_threshold=1.0,
        replace_underscore=True,
        trailing_comma=False,
        exclude_tags=DEFAULT_EXCLUDE_TAGS,
    )

    assert segs is service.result.segs
    assert positive is service.result.positive
    assert service.call["image"] is image
    assert service.call["clip"] == "clip"
    assert service.call["wd14_tagger"] is wd14_tagger
    assert service.call["universal_positive"] == "masterpiece"
    tile_controls = cast(TileSEGSControls, service.call["tile_controls"])
    tag_controls = cast(WD14TagFormattingControls, service.call["tag_controls"])
    assert tile_controls.bbox_size == 872
    assert tag_controls.threshold == 0.35


class _FakeService:
    """Capture node calls for delegation tests."""

    def __init__(self) -> None:
        """Create a fake service result."""

        segment = Segment(
            cropped_image=None,
            cropped_mask=torch.ones((8, 8)),
            confidence=1.0,
            crop_region=CropRegion(0, 0, 8, 8),
            bbox=BoundingBox(0, 0, 8, 8),
            label="tile_001",
        )
        self.result = TileAndTagSEGSResult(
            segs=((8, 8), [segment]),
            positive=ConditioningBatch(("encoded",)),
        )
        self.call: dict[str, object] = {}

    def tile_and_tag(
        self,
        **kwargs: object,
    ) -> TileAndTagSEGSResult:
        """Return a fixed result and remember provided inputs."""

        self.call = kwargs
        return self.result
