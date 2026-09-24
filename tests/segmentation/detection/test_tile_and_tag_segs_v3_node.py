# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Tile & Tag SEGS Comfy v3 wrapper."""

from __future__ import annotations

from typing import Any

import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.segs import BoundingBox, CropRegion, Segment
from simple_syrup.nodes.tile_and_tag_segs import TileAndTagSEGS
from simple_syrup.nodes_v3.tile_and_tag_segs import TileAndTagSEGSV3
from simple_syrup.services.tile_and_tag_segs_service import TileAndTagSEGSResult


def test_tile_and_tag_segs_v3_schema_includes_wd14_tagger() -> None:
    """The v3 schema mirrors the legacy loaded-WD14 node contract."""

    schema = TileAndTagSEGSV3.define_schema()

    assert schema.node_id == "SimpleSyrup.TileAndTagSEGS"
    assert schema.display_name == "Tile & Tag SEGS"
    assert [input_item.id for input_item in schema.inputs][:3] == [
        "image",
        "clip",
        "wd14_tagger",
    ]
    assert schema.inputs[2].io_type == "WD14_TAGGER"
    universal_positive = schema.inputs[3]
    assert universal_positive.io_type == "STRING"
    assert universal_positive.default == ""
    assert universal_positive.multiline is False
    assert "model" not in [input_item.id for input_item in schema.inputs]
    assert [output.id for output in schema.outputs] == ["segs", "positive"]
    assert [output.io_type for output in schema.outputs] == [
        "SEGS",
        "CONDITIONING_BATCH",
    ]


def test_tile_and_tag_segs_v3_execute_forwards_universal_positive(
    monkeypatch: Any,
) -> None:
    """The v3 wrapper forwards universal_positive to the legacy implementation."""

    service = _FakeService()
    monkeypatch.setattr(TileAndTagSEGS, "service_class", lambda: service)
    image = torch.zeros((1, 8, 8, 3))
    wd14_tagger = object()

    segs, positive = TileAndTagSEGSV3.execute(
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
        exclude_tags="",
    )

    assert segs is service.result.segs
    assert positive is service.result.positive
    assert service.call["wd14_tagger"] is wd14_tagger
    assert service.call["universal_positive"] == "masterpiece"


class _FakeService:
    """Capture v3 wrapper calls through the legacy node."""

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
