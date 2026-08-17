# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute one cold and one warmed SDXL attribution request."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from tools.comfy_api import JsonObject

from .cold_path_results import SdxlColdPathTiming, decode_cold_path_timing
from .cold_path_workflow import BuiltSdxlColdPathWorkflow


class ColdPathClient(Protocol):
    """Submit Comfy workflows and return their completed histories."""

    def submit(self, prompt: dict[str, JsonObject]) -> str:
        """Submit one API-format graph."""

    def wait_for_history(self, prompt_id: str, *, timeout: float) -> JsonObject:
        """Wait for one completed prompt history."""


@dataclass(frozen=True, slots=True)
class SdxlColdPathObservation:
    """Retain the predeclared adjacent cold and warm executions."""

    cold: SdxlColdPathTiming
    warm: SdxlColdPathTiming


def measure_cold_path(
    client: ColdPathClient,
    *,
    workflow: BuiltSdxlColdPathWorkflow,
    first_seed: int,
    run_id: str,
    prompt_timeout: float,
) -> SdxlColdPathObservation:
    """Measure exactly one cache miss followed by one exact warmed reuse."""

    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Cold-path measurement run id must be non-empty.")
    cold = _execute(
        client,
        workflow=workflow,
        seed=first_seed,
        execution_id=f"{run_id}:cold",
        prompt_timeout=prompt_timeout,
    )
    warm = _execute(
        client,
        workflow=workflow,
        seed=first_seed + 1,
        execution_id=f"{run_id}:warm",
        prompt_timeout=prompt_timeout,
    )
    return SdxlColdPathObservation(cold, warm)


def _execute(
    client: ColdPathClient,
    *,
    workflow: BuiltSdxlColdPathWorkflow,
    seed: int,
    execution_id: str,
    prompt_timeout: float,
) -> SdxlColdPathTiming:
    """Submit and decode one synchronized attribution execution."""

    prompt = workflow.prompt_for_execution(seed=seed, run_id=execution_id)
    started_at_ns = time.perf_counter_ns()
    prompt_id = client.submit(prompt)
    history = client.wait_for_history(prompt_id, timeout=prompt_timeout)
    return decode_cold_path_timing(
        history,
        terminal_node_id=workflow.terminal_node_id,
        started_at_ns=started_at_ns,
        seed=seed,
    )
