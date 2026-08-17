# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prime the exact historical dependencies before regional cold attribution."""

from __future__ import annotations

import time
from dataclasses import dataclass

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.visual_case_model import SdxlVisualCase

from .cold_path_measurement import ColdPathClient
from .conventional_variant_workflow import (
    ConventionalVariantBranch,
    build_conventional_variant_steady_state_workflow,
)
from .steady_state_measurement import (
    SdxlSteadyStateTiming,
    decode_steady_state_timing,
)
from .steady_state_workflow import BuiltSdxlSteadyStateWorkflow
from .two_adapter_workflow import (
    TwoAdapterPerformanceMode,
    build_two_adapter_steady_state_workflow,
)


@dataclass(frozen=True, slots=True)
class SdxlColdPathPrimer:
    """Bind one historical primer label to its stable workflow."""

    label: str
    workflow: BuiltSdxlSteadyStateWorkflow


def build_cold_path_primers(
    *,
    checkpoint_name: str,
    mask_names: tuple[str, str],
    case: SdxlVisualCase,
) -> tuple[SdxlColdPathPrimer, ...]:
    """Build the global, left, then right historical preparation order."""

    return (
        SdxlColdPathPrimer(
            TwoAdapterPerformanceMode.GLOBAL_REFERENCE.value,
            build_two_adapter_steady_state_workflow(
                checkpoint_name=checkpoint_name,
                mask_names=mask_names,
                case=case,
                mode=TwoAdapterPerformanceMode.GLOBAL_REFERENCE,
            ),
        ),
        *(
            SdxlColdPathPrimer(
                branch.value,
                build_conventional_variant_steady_state_workflow(
                    checkpoint_name=checkpoint_name,
                    case=case,
                    branch=branch,
                ),
            )
            for branch in ConventionalVariantBranch
        ),
    )


def execute_cold_path_primers(
    client: ColdPathClient,
    *,
    primers: tuple[SdxlColdPathPrimer, ...],
    seed: int,
    prompt_timeout: float,
) -> dict[str, SdxlSteadyStateTiming]:
    """Execute each primer once and return synchronized timing evidence."""

    if tuple(primer.label for primer in primers) != (
        TwoAdapterPerformanceMode.GLOBAL_REFERENCE.value,
        ConventionalVariantBranch.LEFT.value,
        ConventionalVariantBranch.RIGHT.value,
    ):
        raise ValueError("Cold-path primers must preserve global, left, right order.")
    return {
        primer.label: _execute_primer(
            client,
            workflow=primer.workflow,
            seed=seed + index,
            prompt_timeout=prompt_timeout,
        )
        for index, primer in enumerate(primers)
    }


def _execute_primer(
    client: ColdPathClient,
    *,
    workflow: BuiltSdxlSteadyStateWorkflow,
    seed: int,
    prompt_timeout: float,
) -> SdxlSteadyStateTiming:
    """Submit one image-free primer and decode its synchronized terminal."""

    prompt: dict[str, JsonObject] = workflow.prompt_for_seed(seed)
    started_at_ns = time.perf_counter_ns()
    prompt_id = client.submit(prompt)
    history = client.wait_for_history(prompt_id, timeout=prompt_timeout)
    return decode_steady_state_timing(
        history,
        completion_node_id=workflow.completion_node_id,
        started_at_ns=started_at_ns,
        seed=seed,
    )
