# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Tag SEGS w/ WD14 Comfy v3 wrapper."""

from __future__ import annotations

from typing import Any

import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.segs import BoundingBox, CropRegion, ImpactSegs, Segment
from simple_syrup.nodes.tag_segs_with_wd14 import TagSEGSWithWD14
from simple_syrup.nodes_v3.tag_segs_with_wd14 import TagSEGSWithWD14V3
from simple_syrup.services.tag_segs_with_wd14_service import TagSEGSWithWD14Result


def test_tag_segs_with_wd14_v3_schema_includes_clip_and_wd14_tagger() -> None:
    """The v3 schema exposes existing-SEGS WD14 tagging inputs."""

    schema = TagSEGSWithWD14V3.define_schema()

    assert schema.node_id == "SimpleSyrup.TagSEGSWithWD14"
    assert schema.display_name == "Tag SEGS w/ WD14"
    assert [input_item.id for input_item in schema.inputs][:4] == [
        "image",
        "segs",
        "clip",
        "wd14_tagger",
    ]
    assert schema.inputs[1].io_type == "SEGS"
    assert schema.inputs[2].io_type == "CLIP"
    assert schema.inputs[3].io_type == "WD14_TAGGER"
    universal_positive = schema.inputs[4]
    assert universal_positive.io_type == "STRING"
    assert universal_positive.default == ""
    assert universal_positive.multiline is False
    assert [output.id for output in schema.outputs] == ["segs", "positive"]
    assert [output.io_type for output in schema.outputs] == [
        "SEGS",
        "CONDITIONING_BATCH",
    ]


def test_tag_segs_with_wd14_v3_execute_forwards_to_legacy_node(
    monkeypatch: Any,
) -> None:
    """The v3 wrapper forwards execution to the legacy implementation."""

    service = _FakeService()
    monkeypatch.setattr(TagSEGSWithWD14, "service_class", lambda: service)
    image = torch.zeros((1, 8, 8, 3))
    segs: ImpactSegs = ((8, 8), [])
    wd14_tagger = object()

    output_segs, positive = TagSEGSWithWD14V3.execute(
        image=image,
        segs=segs,
        clip="clip",
        wd14_tagger=wd14_tagger,
        universal_positive="masterpiece",
        threshold=0.35,
        character_threshold=1.0,
        replace_underscore=True,
        trailing_comma=False,
        exclude_tags="",
    )

    assert output_segs is service.result.segs
    assert positive is service.result.positive
    assert service.call["segs"] is segs
    assert service.call["clip"] == "clip"
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
            label="seg_001",
        )
        self.result = TagSEGSWithWD14Result(
            segs=((8, 8), [segment]),
            positive=ConditioningBatch(("encoded",)),
        )
        self.call: dict[str, object] = {}

    def tag(self, **kwargs: object) -> TagSEGSWithWD14Result:
        """Return a fixed result and remember provided inputs."""

        self.call = kwargs
        return self.result
