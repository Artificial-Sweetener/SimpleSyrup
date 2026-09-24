# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify benchmark-only regional Anima visual-profile boundaries."""

from __future__ import annotations

import json
from typing import Any

import pytest

from simple_syrup.domain.regional_lora_plan import RegionalLoraBranch
from tools.attention_coupling_benchmark.comfy_probe.anima_regional_profile import (
    BuiltStaticAnimaRegionalProfile,
    _canonical_conditioning_batch,
    _unit_region_strengths,
)
from tools.attention_coupling_benchmark.comfy_probe.anima_regional_profile_node import (
    StaticAnimaRegionalProfileV3,
)
from tools.attention_coupling_benchmark.comfy_probe.anima_regional_profile_spec import (
    decode_visual_regional_adapters,
)


class _ProfileBuilder:
    """Capture exact thin-node delegation for one test."""

    def __init__(self) -> None:
        """Initialize with no observed arguments."""

        self.arguments: dict[str, Any] | None = None

    def build(self, **arguments: Any) -> BuiltStaticAnimaRegionalProfile:
        """Return a recognizable derived model value."""

        self.arguments = arguments
        return BuiltStaticAnimaRegionalProfile(
            "derived-model", "base-positive", "base-negative"
        )


def test_profile_probe_schema_is_dev_only_and_delegates_decoded_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep schema ownership thin while preserving ordered region assignments."""

    builder = _ProfileBuilder()
    monkeypatch.setattr(StaticAnimaRegionalProfileV3, "profile_builder", builder)
    assignments = [
        {
            "region_index": 0,
            "branch": "positive",
            "lora_name": "Anima\\style\\adapter-a.safetensors",
            "strength": 0.8,
        },
        {
            "region_index": 1,
            "branch": "negative",
            "lora_name": "Anima\\style\\adapter-b.safetensors",
            "strength": -0.25,
        },
    ]

    output = StaticAnimaRegionalProfileV3.execute(
        model="model",
        positive="positive",
        negative="negative",
        region_masks="masks",
        latent={"samples": "latent"},
        regional_adapters_json=json.dumps(assignments),
    )

    schema = StaticAnimaRegionalProfileV3.define_schema()
    assert schema.node_id == "SimpleSyrupBenchmark.StaticAnimaRegionalProfile"
    assert schema.is_dev_only is True
    assert output.result == ("derived-model", "base-positive", "base-negative")
    assert builder.arguments is not None
    adapters = builder.arguments["adapters"]
    assert [adapter.lora_name for adapter in adapters] == [
        "Anima\\style\\adapter-a.safetensors",
        "Anima\\style\\adapter-b.safetensors",
    ]
    assert [adapter.branch for adapter in adapters] == [
        RegionalLoraBranch.POSITIVE,
        RegionalLoraBranch.NEGATIVE,
    ]


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ("not-json", ValueError, "valid JSON"),
        ("[]", ValueError, "one to four"),
        (
            json.dumps(
                [
                    {
                        "region_index": 0,
                        "branch": "positive",
                        "lora_name": "adapter.safetensors",
                    }
                ]
            ),
            ValueError,
            "exactly",
        ),
        (
            json.dumps(
                [
                    {
                        "region_index": 0,
                        "branch": "global",
                        "lora_name": "adapter.safetensors",
                        "strength": 1.0,
                    }
                ]
            ),
            ValueError,
            "positive or negative",
        ),
    ],
)
def test_visual_adapter_spec_fails_closed(
    value: str,
    error: type[Exception],
    message: str,
) -> None:
    """Reject malformed or ambiguous benchmark assignment records."""

    with pytest.raises(error, match=message):
        decode_visual_regional_adapters(value)


def test_probe_adapts_only_valid_foreign_conditioning_batches() -> None:
    """Cross the dev-extension package identity without weakening batch shape."""

    foreign = type("ForeignConditioningBatch", (), {"entries": (["context", {}],)})()

    canonical = _canonical_conditioning_batch(foreign, "positive")

    assert canonical.entries == (["context", {}],)
    with pytest.raises(TypeError, match="non-empty ConditioningBatch"):
        _canonical_conditioning_batch([], "negative")


def test_visual_profile_enables_every_canonical_region() -> None:
    """Keep split-mask execution strengths aligned with both regional masks."""

    assert _unit_region_strengths(2) == (1.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        _unit_region_strengths(0)
