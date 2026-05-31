# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Batch Region Conditioning Comfy v3 wrapper."""

from __future__ import annotations

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.nodes_v3.batch_region_conditioning import (
    BatchRegionConditioningV3,
)


def test_batch_region_conditioning_v3_schema_uses_mixed_autogrow_inputs() -> None:
    """The v3 schema accepts conditioning values and conditioning batches."""

    schema = BatchRegionConditioningV3.define_schema()

    assert schema.node_id == "SimpleSyrup.BatchRegionConditioning"
    assert schema.display_name == "Batch Region Conditioning"
    assert schema.category == "SimpleSyrup/Conditioning"
    assert [input_item.id for input_item in schema.inputs] == ["conditioning_inputs"]
    assert schema.inputs[0].io_type == "COMFY_AUTOGROW_V3"
    assert schema.inputs[0].template.prefix == "conditioning"
    assert schema.inputs[0].template.min == 2
    assert schema.inputs[0].template.max == 50
    assert schema.inputs[0].template.input.get_io_type() == (
        "CONDITIONING,CONDITIONING_BATCH"
    )
    assert [output.id for output in schema.outputs] == ["batch"]
    assert schema.outputs[0].io_type == "CONDITIONING_BATCH"


def test_batch_region_conditioning_v3_execute_flattens_inputs() -> None:
    """The v3 wrapper batches mixed inputs in Autogrow insertion order."""

    auto = ConditioningBatch(("auto 1", "auto 2"))
    hand = "hand 1"
    extra = ConditioningBatch(("auto 3",))

    (batch,) = BatchRegionConditioningV3.execute(
        {
            "conditioning0": auto,
            "conditioning1": hand,
            "conditioning2": extra,
        }
    )

    assert batch == ConditioningBatch(("auto 1", "auto 2", "hand 1", "auto 3"))
