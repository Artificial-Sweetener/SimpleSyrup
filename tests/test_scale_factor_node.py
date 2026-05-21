# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Scale Factor node contract."""

from __future__ import annotations

from simple_syrup.nodes import tooltips
from simple_syrup.nodes.scale_factor import ScaleFactor, scale_factor_options


def test_scale_factor_options_returns_fresh_bounded_metadata() -> None:
    """Scale-factor widget options expose bounded FLOAT metadata."""

    first_options = scale_factor_options()
    second_options = scale_factor_options(default=1.0)

    assert first_options is not second_options
    assert first_options["default"] == 1.5
    assert first_options["min"] == 1.0
    assert first_options["max"] == 5.0
    assert first_options["step"] == 0.1
    assert isinstance(first_options["tooltip"], str)
    assert first_options["tooltip"]
    assert second_options["default"] == 1.0


def test_scale_factor_node_contract() -> None:
    """Scale Factor exposes a bounded FLOAT primitive contract."""

    inputs = ScaleFactor.INPUT_TYPES()
    value_type, value_options = inputs["required"]["value"]

    assert ScaleFactor.RETURN_TYPES == ("FLOAT",)
    assert ScaleFactor.RETURN_NAMES == ("scale_factor",)
    assert ScaleFactor.FUNCTION == "get_value"
    assert ScaleFactor.CATEGORY == "SimpleSyrup/Primitives"
    assert ScaleFactor.DESCRIPTION == "Provides a bounded multiplier for scaling."
    assert len(ScaleFactor.OUTPUT_TOOLTIPS) == 1
    assert ScaleFactor.OUTPUT_TOOLTIPS[0] == (
        "Multiplier used to scale a connected target."
    )
    assert list(inputs["required"]) == ["value"]
    assert value_type == "FLOAT"
    assert value_options["default"] == 1.5
    assert value_options["min"] == 1.0
    assert value_options["max"] == 5.0
    assert value_options["step"] == 0.1
    assert value_options["tooltip"] == tooltips.SCALE_FACTOR_VALUE


def test_scale_factor_node_returns_float_value() -> None:
    """Scale Factor returns the provided value as a FLOAT output."""

    assert ScaleFactor().get_value("2.5") == (2.5,)
