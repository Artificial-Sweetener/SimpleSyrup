# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the shared seed-only steady-state measurement owner."""

from __future__ import annotations

import time

import pytest

from tools.comfy_api import JsonObject
from tools.sdxl_regional_lora_performance.steady_state_runner import (
    measure_steady_state_workflow,
)
from tools.sdxl_regional_lora_performance.steady_state_workflow import (
    BuiltSdxlSteadyStateWorkflow,
)


class _Client:
    """Return synchronized completion histories while recording submitted seeds."""

    def __init__(self) -> None:
        """Initialize the recorded seed sequence."""

        self.seeds: list[int] = []

    def submit(self, prompt: dict[str, JsonObject]) -> str:
        """Record one sampler seed and return a stable prompt identifier."""

        inputs = prompt["1"]["inputs"]
        if not isinstance(inputs, dict):
            raise TypeError("Test sampler inputs must be an object.")
        seed = inputs["seed"]
        if not isinstance(seed, int):
            raise TypeError("Test sampler seed must be an integer.")
        self.seeds.append(seed)
        return f"prompt-{len(self.seeds)}"

    def wait_for_history(self, prompt_id: str, *, timeout: float) -> JsonObject:
        """Return one completion after the caller's synchronized start."""

        assert prompt_id
        assert timeout == 12.0
        return {
            "outputs": {
                "2": {
                    "benchmark_completion": [
                        {"completed_at_ns": time.perf_counter_ns() + 1_000_000}
                    ]
                }
            }
        }


def test_measurement_warms_once_then_records_five_seed_only_revisions() -> None:
    """Keep the prepared graph stable and advance only the sampler seed."""

    client = _Client()
    observation = measure_steady_state_workflow(
        client,
        workflow=_workflow(),
        repeats=5,
        first_seed=100,
        prompt_timeout=12.0,
    )

    assert client.seeds == [100, 101, 102, 103, 104, 105]
    assert observation.warmup.seed == 100
    assert tuple(value.seed for value in observation.measured) == (
        101,
        102,
        103,
        104,
        105,
    )
    assert observation.median_runtime_ms > 0.0


def test_measurement_rejects_less_than_five_repeats() -> None:
    """Enforce the RA-07 warmed timing sample size at the shared owner."""

    with pytest.raises(ValueError, match="at least five repeats"):
        measure_steady_state_workflow(
            _Client(),
            workflow=_workflow(),
            repeats=4,
            first_seed=100,
            prompt_timeout=12.0,
        )


def _workflow() -> BuiltSdxlSteadyStateWorkflow:
    """Build one minimal generic seed-revisable graph."""

    return BuiltSdxlSteadyStateWorkflow(
        prompt={
            "1": {"class_type": "KSampler", "inputs": {"seed": 0}},
            "2": {
                "class_type": "SimpleSyrupBenchmark.CompleteLatent",
                "inputs": {"latent": ["1", 0]},
            },
        },
        sampler_node_id="1",
        completion_node_id="2",
    )
