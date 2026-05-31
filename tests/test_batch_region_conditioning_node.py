# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Batch Region Conditioning legacy node."""

from __future__ import annotations

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.nodes.batch_region_conditioning import BatchRegionConditioning


def test_batch_region_conditioning_contract() -> None:
    """Batch Region Conditioning exposes a mixed-input legacy contract."""

    inputs = BatchRegionConditioning.INPUT_TYPES()

    assert BatchRegionConditioning.RETURN_TYPES == ("CONDITIONING_BATCH",)
    assert BatchRegionConditioning.RETURN_NAMES == ("batch",)
    assert BatchRegionConditioning.FUNCTION == "batch"
    assert BatchRegionConditioning.CATEGORY == "SimpleSyrup/Conditioning"
    assert list(inputs["required"]) == ["first", "second"]
    assert inputs["required"]["first"][0] == "CONDITIONING,CONDITIONING_BATCH"
    assert inputs["required"]["second"][0] == "CONDITIONING,CONDITIONING_BATCH"


def test_batch_region_conditioning_node_flattens_mixed_inputs() -> None:
    """The legacy node flattens batches and normal conditionings in order."""

    auto = ConditioningBatch(("auto 1", "auto 2"))
    hand = "hand 1"

    (batch,) = BatchRegionConditioning().batch(auto, hand)

    assert batch == ConditioningBatch(("auto 1", "auto 2", "hand 1"))
