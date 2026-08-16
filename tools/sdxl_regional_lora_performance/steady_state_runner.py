# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Measure one prepared SDXL workflow through seed-only revisions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from tools.comfy_api import JsonObject

from .steady_state_measurement import (
    SdxlSteadyStateTiming,
    decode_steady_state_timing,
    median_steady_state_runtime_ms,
)
from .steady_state_workflow import BuiltSdxlSteadyStateWorkflow


class SdxlSteadyStateClient(Protocol):
    """Submit Comfy prompts and wait for their completed history."""

    def submit(self, prompt: dict[str, JsonObject]) -> str:
        """Submit one API-format prompt graph."""

    def wait_for_history(self, prompt_id: str, *, timeout: float) -> JsonObject:
        """Wait for one prompt to finish and return its history."""


@dataclass(frozen=True, slots=True)
class SdxlSteadyStateObservation:
    """Retain one warmup and the accepted measured timing sample."""

    warmup: SdxlSteadyStateTiming
    measured: tuple[SdxlSteadyStateTiming, ...]
    median_runtime_ms: float


def measure_steady_state_workflow(
    client: SdxlSteadyStateClient,
    *,
    workflow: BuiltSdxlSteadyStateWorkflow,
    repeats: int,
    first_seed: int,
    prompt_timeout: float,
) -> SdxlSteadyStateObservation:
    """Warm once and measure at least five seed-only graph revisions."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 5:
        raise ValueError("Steady-state timing requires at least five repeats.")
    warmup = _execute_once(
        client,
        workflow=workflow,
        seed=first_seed,
        prompt_timeout=prompt_timeout,
    )
    measured = tuple(
        _execute_once(
            client,
            workflow=workflow,
            seed=first_seed + index + 1,
            prompt_timeout=prompt_timeout,
        )
        for index in range(repeats)
    )
    return SdxlSteadyStateObservation(
        warmup=warmup,
        measured=measured,
        median_runtime_ms=median_steady_state_runtime_ms(measured),
    )


def _execute_once(
    client: SdxlSteadyStateClient,
    *,
    workflow: BuiltSdxlSteadyStateWorkflow,
    seed: int,
    prompt_timeout: float,
) -> SdxlSteadyStateTiming:
    """Submit one seed-only graph revision and decode synchronized wall time."""

    prompt = workflow.prompt_for_seed(seed)
    started_at_ns = time.perf_counter_ns()
    prompt_id = client.submit(prompt)
    history = client.wait_for_history(prompt_id, timeout=prompt_timeout)
    return decode_steady_state_timing(
        history,
        completion_node_id=workflow.completion_node_id,
        started_at_ns=started_at_ns,
        seed=seed,
    )
