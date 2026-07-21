# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the standard regional-conditioning composer node."""

from __future__ import annotations

from typing import ClassVar

import torch

from simple_syrup.nodes_v3.compose_regional_conditioning import (
    ComposeRegionalConditioningV3,
)


class FakeConditioningService:
    """Record one composition request and return recognizable outputs."""

    calls: ClassVar[list[dict[str, object]]] = []

    def assemble(self, **kwargs: object) -> tuple[object, object]:
        """Record arguments and return standard-conditioning stand-ins."""

        type(self).calls.append(kwargs)
        return "positive-conditioning", "negative-conditioning"


def test_composer_schema_outputs_native_conditioning() -> None:
    """The composer exposes the same regional inputs and standard outputs."""

    schema = ComposeRegionalConditioningV3.define_schema()

    assert schema.node_id == "SimpleSyrup.ComposeRegionalConditioning"
    assert schema.display_name == "Compose Regional Conditioning"
    assert schema.category == "SimpleSyrup/Conditioning"
    assert [input_item.id for input_item in schema.inputs] == [
        "positive",
        "negative",
        "region_masks",
        "regional_prompt_weight",
        "region_mask_feather",
    ]
    assert schema.inputs[3].default == 0.5
    assert [output.io_type for output in schema.outputs] == [
        "CONDITIONING",
        "CONDITIONING",
    ]


def test_composer_delegates_to_authoritative_regional_service() -> None:
    """The node adds no policy beyond its application service."""

    original = ComposeRegionalConditioningV3.conditioning_service_class
    ComposeRegionalConditioningV3.conditioning_service_class = FakeConditioningService  # type: ignore[assignment]
    FakeConditioningService.calls = []
    masks = torch.ones((2, 4, 4))
    try:
        output = ComposeRegionalConditioningV3.execute(
            positive="positive-batch",
            negative="negative-batch",
            region_masks=masks,
            regional_prompt_weight=0.75,
            region_mask_feather=3,
        )
    finally:
        ComposeRegionalConditioningV3.conditioning_service_class = original

    assert output == ("positive-conditioning", "negative-conditioning")
    assert FakeConditioningService.calls == [
        {
            "positive": "positive-batch",
            "negative": "negative-batch",
            "masks": masks,
            "regional_prompt_weight": 0.75,
            "region_mask_feather": 3,
        }
    ]
