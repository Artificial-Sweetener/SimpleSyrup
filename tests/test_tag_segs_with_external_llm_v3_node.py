# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Tag SEGS w/ External LLM Comfy v3 node."""

from __future__ import annotations

from typing import Any

import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.segs import BoundingBox, CropRegion, ImpactSegs, Segment
from simple_syrup.nodes_v3.tag_segs_with_external_llm import (
    TagSEGSWithExternalLLMV3,
)
from simple_syrup.services.tag_segs_with_external_llm_service import (
    LLMTagFormattingControls,
    TagSEGSWithExternalLLMResult,
)


def test_tag_segs_with_external_llm_v3_schema(monkeypatch: Any) -> None:
    """The v3 schema exposes the external LLM SEGS tagging contract."""

    monkeypatch.setattr(TagSEGSWithExternalLLMV3, "_service", _FakeService())

    schema = TagSEGSWithExternalLLMV3.define_schema()

    assert schema.node_id == "SimpleSyrup.TagSEGSWithExternalLLM"
    assert schema.display_name == "Tag SEGS w/ External LLM"
    assert schema.category == "SimpleSyrup/Detailing"
    assert [input_item.id for input_item in schema.inputs] == [
        "image",
        "segs",
        "clip",
        "model",
        "system_prompt",
        "user_prompt",
        "universal_positive",
        "seg_image_mode",
        "replace_underscore",
        "trailing_comma",
        "exclude_tags",
        "max_tokens",
        "reasoning_effort",
    ]
    assert schema.inputs[0].io_type == "IMAGE"
    assert schema.inputs[1].io_type == "SEGS"
    assert schema.inputs[2].io_type == "CLIP"
    assert schema.inputs[3].options == ["vision-model"]
    assert schema.inputs[4].default == ""
    assert schema.inputs[5].default == ""
    seg_image_mode = schema.inputs[7]
    assert seg_image_mode.io_type == "COMBO"
    assert seg_image_mode.options == ["transparent mask", "black mask", "full crop"]
    assert seg_image_mode.default == "transparent mask"
    assert schema.inputs[8].default is True
    assert schema.inputs[9].default is False
    assert schema.inputs[10].default == ""
    assert schema.inputs[11].default == 1024
    assert schema.inputs[12].options == ["default", "high", "medium", "low", "off"]
    assert [output.id for output in schema.outputs] == ["segs", "positive"]
    assert [output.io_type for output in schema.outputs] == [
        "SEGS",
        "CONDITIONING_BATCH",
    ]


def test_tag_segs_with_external_llm_v3_execute_forwards_to_service(
    monkeypatch: Any,
) -> None:
    """The v3 node delegates execution to the service."""

    service = _FakeService()
    monkeypatch.setattr(TagSEGSWithExternalLLMV3, "_service", service)
    image = torch.zeros((1, 8, 8, 3))
    segs: ImpactSegs = ((8, 8), [])

    output_segs, positive = TagSEGSWithExternalLLMV3.execute(
        image=image,
        segs=segs,
        clip="clip",
        model="vision-model",
        system_prompt="system",
        user_prompt="user",
        universal_positive="masterpiece",
        seg_image_mode="black mask",
        replace_underscore=False,
        trailing_comma=True,
        exclude_tags="bad tag",
        max_tokens=128,
        reasoning_effort="off",
    )

    assert output_segs is service.result.segs
    assert positive is service.result.positive
    assert service.call["image"] is image
    assert service.call["segs"] is segs
    assert service.call["clip"] == "clip"
    assert service.call["model"] == "vision-model"
    assert service.call["system_prompt"] == "system"
    assert service.call["user_prompt"] == "user"
    assert service.call["universal_positive"] == "masterpiece"
    assert service.call["seg_image_mode"] == "black mask"
    assert service.call["max_tokens"] == 128
    assert service.call["reasoning_effort"] == "off"
    formatting = service.call["formatting"]
    assert isinstance(formatting, LLMTagFormattingControls)
    assert formatting.replace_underscore is False
    assert formatting.trailing_comma is True
    assert formatting.exclude_tags == "bad tag"


class _FakeService:
    """Capture v3 node service calls."""

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
        self.result = TagSEGSWithExternalLLMResult(
            segs=((8, 8), [segment]),
            positive=ConditioningBatch(("encoded",)),
        )
        self.call: dict[str, object] = {}

    def model_choices(self) -> list[str]:
        """Return deterministic model choices."""

        return ["vision-model"]

    def tag(self, **kwargs: object) -> TagSEGSWithExternalLLMResult:
        """Return a fixed result and remember provided inputs."""

        self.call = kwargs
        return self.result
