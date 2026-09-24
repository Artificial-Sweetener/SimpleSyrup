# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Scale Factor Comfy v3 wrapper."""

from __future__ import annotations

from simple_syrup.nodes import tooltips
from simple_syrup.nodes_v3.scale_factor import ScaleFactorV3


def test_scale_factor_v3_schema() -> None:
    """Scale Factor v3 schema exposes the same bounded float contract."""

    schema = ScaleFactorV3.define_schema()

    assert schema.node_id == "SimpleSyrup.ScaleFactor"
    assert schema.display_name == "Scale Factor"
    assert schema.category == "SimpleSyrup/Primitives"
    assert schema.description == "Provides a bounded multiplier for scaling."
    assert [input_item.id for input_item in schema.inputs] == ["value"]
    assert schema.inputs[0].io_type == "FLOAT"
    assert schema.inputs[0].default == 1.5
    assert schema.inputs[0].min == 1.0
    assert schema.inputs[0].max == 5.0
    assert schema.inputs[0].step == 0.1
    assert schema.inputs[0].tooltip == tooltips.SCALE_FACTOR_VALUE
    assert [output.id for output in schema.outputs] == ["scale_factor"]
    assert schema.outputs[0].io_type == "FLOAT"
    assert schema.outputs[0].tooltip == "Multiplier used to scale a connected target."


def test_scale_factor_v3_execute_delegates_to_legacy_node() -> None:
    """Scale Factor v3 execution returns the legacy node output shape."""

    assert ScaleFactorV3.execute(2.5) == (2.5,)
