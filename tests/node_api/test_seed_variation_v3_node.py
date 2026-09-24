# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Seed Variation Comfy v3 MODEL patch node."""

from __future__ import annotations

import pytest

from simple_syrup.nodes import tooltips
from simple_syrup.nodes_v3 import seed_variation
from simple_syrup.nodes_v3.seed_variation import SeedVariationV3


def test_seed_variation_v3_schema_exposes_complete_model_patch_contract() -> None:
    """Pin the public id, controls, descriptions, and MODEL output metadata."""

    schema = SeedVariationV3.define_schema()

    assert schema.node_id == "SimpleSyrup.SeedVariation"
    assert schema.display_name == "Seed Variation"
    assert schema.category == "SimpleSyrup/Sampling"
    assert "mixing sampler noise" in schema.description
    assert [item.id for item in schema.inputs] == [
        "model",
        "variation_seed",
        "variation_strength",
    ]
    model_input, seed_input, strength_input = schema.inputs
    assert model_input.io_type == "MODEL"
    assert model_input.tooltip == tooltips.SEED_VARIATION_MODEL_INPUT
    assert seed_input.io_type == "INT"
    assert seed_input.default == 0
    assert seed_input.min == 0
    assert seed_input.max == 0xFFFFFFFFFFFFFFFF
    assert seed_input.control_after_generate is True
    assert seed_input.tooltip == tooltips.VARIATION_SEED
    assert strength_input.io_type == "FLOAT"
    assert strength_input.default == 0.0
    assert strength_input.min == 0.0
    assert strength_input.max == 1.0
    assert strength_input.step == 0.01
    assert strength_input.tooltip == tooltips.VARIATION_STRENGTH
    assert [output.id for output in schema.outputs] == ["model"]
    assert schema.outputs[0].io_type == "MODEL"
    assert schema.outputs[0].tooltip == tooltips.SEED_VARIATION_MODEL_OUTPUT


def test_seed_variation_v3_execute_delegates_to_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the Comfy-facing node thin and return the service result exactly."""

    source = object()
    derived = object()
    calls: list[dict[str, object]] = []

    class RecordingService:
        """Record one node-to-service request."""

        def prepare(self, **kwargs: object) -> object:
            """Capture keyword arguments and return a stable MODEL sentinel."""

            calls.append(kwargs)
            return derived

    monkeypatch.setattr(
        seed_variation,
        "SEED_VARIATION_MODEL_SERVICE",
        RecordingService(),
    )

    result = SeedVariationV3.execute(source, 83, 0.25)

    assert result == (derived,)
    assert calls == [
        {
            "model": source,
            "variation_seed": 83,
            "variation_strength": 0.25,
        }
    ]


def test_seed_variation_v3_execute_surfaces_invalid_settings() -> None:
    """Expose actionable domain validation before a MODEL can be mutated."""

    with pytest.raises(ValueError, match="Variation strength must be between"):
        SeedVariationV3.execute(object(), 1, 1.5)
