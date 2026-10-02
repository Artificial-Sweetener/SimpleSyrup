# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Protect the Extras sampler's schema and inversion-free execution contract."""

from __future__ import annotations

from inspect import signature
from typing import Any

import pytest

from simple_syrup.nodes.ksampler_extras import KSamplerExtras
from simple_syrup.nodes_v3.legacy_node_wrappers import KSamplerExtrasV3
from simple_syrup.nodes_v3.legacy_workflow_input_order import (
    KSAMPLER_EXTRAS_INPUT_ORDER,
)
from simple_syrup.services.ksampler_sampling_service import KSamplerSamplingService


def test_extras_schema_has_only_ordinary_sampling_inputs() -> None:
    """Keep persisted input order without adding inversion widgets."""
    schema = KSamplerExtrasV3.define_schema()
    assert schema.node_id == "SimpleSyrup.KSamplerExtras"
    assert tuple(item.id for item in schema.inputs) == KSAMPLER_EXTRAS_INPUT_ORDER
    negative = next(item for item in schema.inputs if item.id == "negative")
    assert negative.optional
    assert "noise_inversion" not in signature(KSamplerExtras.sample).parameters


def test_extras_v3_execution_does_not_select_inversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate ordinary sampling without injecting the default inversion recipe."""
    received: dict[str, Any] = {}
    latent: dict[str, Any] = {"samples": object()}

    class RecordingSamplingService(KSamplerSamplingService):
        """Record the node-to-service boundary without neural execution."""

        def sample(self, **kwargs: Any) -> dict[str, Any]:
            """Capture delegated inputs and return the original latent."""
            received.update(kwargs)
            return latent

    monkeypatch.setattr(KSamplerExtras, "service_class", RecordingSamplingService)
    result = KSamplerExtrasV3.execute(
        model=object(),
        seed=123,
        steps=4,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive=object(),
        latent_image=latent,
    )
    assert result == (latent,)
    assert received["latent_image"] is latent
    assert received["negative"] is None
    assert received["seed"] == 123
    assert "noise_inversion" not in received
