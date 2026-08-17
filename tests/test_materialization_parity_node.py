# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the benchmark-only materialization-parity node boundary."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from tools.attention_coupling_benchmark.comfy_probe import comfy_entrypoint
from tools.attention_coupling_benchmark.comfy_probe.materialization_parity_node import (
    CompareMaterializationParityV3,
)


@dataclass(frozen=True)
class _Observation:
    """Return one stable fake probe payload."""

    def as_json_object(self) -> dict[str, object]:
        """Return exact fake numerical evidence."""

        return {"exact": True, "element_count": 4}


class _Probe:
    """Capture the node-to-probe call without model execution."""

    def __init__(self) -> None:
        """Initialize empty captured keyword arguments."""

        self.arguments: dict[str, object] | None = None

    def compare(self, **arguments: object) -> _Observation:
        """Retain all arguments and return one fixed observation."""

        self.arguments = arguments
        return _Observation()


def test_node_delegates_exact_inputs_and_publishes_one_ui_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep graph adaptation in the probe and JSON publication in the node."""

    probe = _Probe()
    monkeypatch.setattr(CompareMaterializationParityV3, "probe", probe)
    values = {
        "model": object(),
        "positive": object(),
        "negative": object(),
        "region_masks": object(),
        "latent_image": object(),
        "region_mask_feather": 12,
    }

    result = CompareMaterializationParityV3.execute(
        model=values["model"],
        positive=values["positive"],
        negative=values["negative"],
        region_masks=values["region_masks"],
        latent_image=values["latent_image"],
        region_mask_feather=12,
        run_id="parity-1",
    )

    assert probe.arguments == values
    assert result.ui["materialization_parity"] == [
        {"exact": True, "element_count": 4, "run_id": "parity-1"}
    ]
    assert '"run_id":"parity-1"' in result.result[0]


def test_node_schema_is_registered_only_for_benchmark_execution() -> None:
    """Expose the focused terminal through the dev-only probe extension."""

    schema = CompareMaterializationParityV3.define_schema()
    extension = asyncio.run(comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())

    assert schema.node_id == "SimpleSyrupBenchmark.CompareMaterializationParity"
    assert schema.is_output_node is True
    assert schema.is_dev_only is True
    assert tuple(item.io_type for item in schema.inputs[1].io_types) == (
        "CONDITIONING",
        "CONDITIONING_BATCH",
    )
    assert tuple(item.io_type for item in schema.inputs[2].io_types) == (
        "CONDITIONING",
        "CONDITIONING_BATCH",
    )
    assert CompareMaterializationParityV3 in nodes
