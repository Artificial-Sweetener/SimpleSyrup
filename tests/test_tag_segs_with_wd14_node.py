# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Tag SEGS w/ WD14 node contract."""

from __future__ import annotations

from typing import Any, cast

import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.segs import BoundingBox, CropRegion, ImpactSegs, Segment
from simple_syrup.nodes.tag_segs_with_wd14 import TagSEGSWithWD14
from simple_syrup.nodes.tile_and_tag_segs import DEFAULT_EXCLUDE_TAGS
from simple_syrup.runtime.wd14_tagger import WD14TagFormattingControls
from simple_syrup.services.tag_segs_with_wd14_service import TagSEGSWithWD14Result


def test_tag_segs_with_wd14_contract() -> None:
    """Tag SEGS w/ WD14 exposes the agreed ComfyUI contract."""

    inputs = TagSEGSWithWD14.INPUT_TYPES()

    assert TagSEGSWithWD14.RETURN_TYPES == ("SEGS", "CONDITIONING_BATCH")
    assert TagSEGSWithWD14.RETURN_NAMES == ("segs", "positive")
    assert TagSEGSWithWD14.FUNCTION == "tag"
    assert TagSEGSWithWD14.CATEGORY == "SimpleSyrup/Detailing"
    assert list(inputs["required"]) == [
        "image",
        "segs",
        "clip",
        "wd14_tagger",
        "universal_positive",
        "threshold",
        "character_threshold",
        "replace_underscore",
        "trailing_comma",
        "exclude_tags",
    ]
    assert inputs["required"]["segs"][0] == "SEGS"
    assert inputs["required"]["clip"][0] == "CLIP"
    assert inputs["required"]["wd14_tagger"][0] == "WD14_TAGGER"
    assert inputs["required"]["universal_positive"][0] == "STRING"
    assert inputs["required"]["universal_positive"][1]["default"] == ""
    assert inputs["required"]["threshold"][1]["default"] == 0.35
    assert inputs["required"]["character_threshold"][1]["default"] == 1.0
    assert inputs["required"]["replace_underscore"][1]["default"] is True
    assert inputs["required"]["trailing_comma"][1]["default"] is False
    assert inputs["required"]["exclude_tags"][1]["default"] == DEFAULT_EXCLUDE_TAGS
    assert "optional" not in inputs


def test_tag_segs_with_wd14_delegates_to_service(monkeypatch: Any) -> None:
    """The node delegates behavior and returns service outputs unchanged."""

    service = _FakeService()
    monkeypatch.setattr(TagSEGSWithWD14, "service_class", lambda: service)
    image = torch.zeros((1, 8, 8, 3))
    segs: ImpactSegs = ((8, 8), [])
    wd14_tagger = object()

    output_segs, positive = TagSEGSWithWD14().tag(
        image=image,
        segs=segs,
        clip="clip",
        wd14_tagger=wd14_tagger,
        universal_positive="masterpiece",
        threshold=0.35,
        character_threshold=1.0,
        replace_underscore=True,
        trailing_comma=False,
        exclude_tags=DEFAULT_EXCLUDE_TAGS,
    )

    assert output_segs is service.result.segs
    assert positive is service.result.positive
    assert service.call["image"] is image
    assert service.call["segs"] is segs
    assert service.call["clip"] == "clip"
    assert service.call["wd14_tagger"] is wd14_tagger
    assert service.call["universal_positive"] == "masterpiece"
    tag_controls = cast(WD14TagFormattingControls, service.call["tag_controls"])
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
