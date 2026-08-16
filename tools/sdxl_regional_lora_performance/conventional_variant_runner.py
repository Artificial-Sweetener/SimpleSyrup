# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Measure the conventional two-variant SDXL regional performance floor."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TypeAlias

from tools.comfy_api import JsonObject, LoopbackComfyClient
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    CHECKPOINT_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import SdxlVisualCase
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_attention_coupling_integration.visual_launch import (
    sdxl_visual_sampling_launch_arguments,
)
from tools.sdxl_attention_coupling_integration.visual_matrix_execution import (
    build_sdxl_visual_model_links,
)

from .conventional_variant_workflow import (
    BuiltConventionalVariantWorkflow,
    ConventionalVariantBranch,
    build_conventional_variant_workflow,
)
from .measurement import SdxlRegionalLoraTiming, decode_timing, median_runtime_ms
from .two_adapter_workflow import (
    BuiltTwoAdapterPerformanceWorkflow,
    TwoAdapterPerformanceMode,
    build_two_adapter_performance_workflow,
)

LOGGER = logging.getLogger(__name__)
TimingWorkflow: TypeAlias = (
    BuiltConventionalVariantWorkflow | BuiltTwoAdapterPerformanceWorkflow
)
WorkflowBuilder: TypeAlias = Callable[[str], TimingWorkflow]


def run_conventional_variant_floor(
    artifacts: IntegrationArtifacts,
    *,
    inventory: SdxlVisualInventory,
    case: SdxlVisualCase,
    comfy_root: Path,
    repeats: int = 3,
) -> Path:
    """Measure one two-LoRA control and two independent conventional variants."""

    if repeats < 3:
        raise ValueError("Conventional variant floor requires three repeats.")
    links = build_sdxl_visual_model_links(comfy_root, inventory)
    observations: dict[str, dict[str, object]] = {}
    system_stats: JsonObject = {}
    server_cleanup = False
    with links:
        preview: list[TimingWorkflow] = [
            build_two_adapter_performance_workflow(
                run_id=f"{artifacts.run_id}:preview:global",
                checkpoint_name=CHECKPOINT_SELECTION,
                mask_names=("unused-left", "unused-right"),
                case=case,
                mode=TwoAdapterPerformanceMode.GLOBAL_REFERENCE,
            )
        ]
        preview.extend(
            tuple(
                build_conventional_variant_workflow(
                    run_id=f"{artifacts.run_id}:preview:{branch.value}",
                    checkpoint_name=CHECKPOINT_SELECTION,
                    case=case,
                    branch=branch,
                )
                for branch in ConventionalVariantBranch
            )
        )
        required = frozenset(
            node_id for workflow in preview for node_id in workflow.required_node_ids
        )
        with ManagedComfyServer(
            comfy_root=comfy_root,
            artifacts=artifacts,
            required_node_ids=required,
            readiness_timeout=240.0,
            launch_arguments=sdxl_visual_sampling_launch_arguments(),
        ) as running:
            system_stats = running.system_stats
            control_key = TwoAdapterPerformanceMode.GLOBAL_REFERENCE.value
            observations[control_key] = _measure(
                running.client,
                label=control_key,
                repeats=repeats,
                build=lambda run_id: _global_workflow(run_id, case),
            )
            LOGGER.info("Completed %s timing block", control_key)
            for branch in ConventionalVariantBranch:
                observations[branch.value] = _measure(
                    running.client,
                    label=branch.value,
                    repeats=repeats,
                    build=_variant_builder(case, branch),
                )
                LOGGER.info("Completed %s timing block", branch.value)
            port = running.port
            process = running.process
        server_cleanup = not process.is_running and is_loopback_port_available(port)
        artifacts.record_cleanup(
            process_running=process.is_running,
            port_available=is_loopback_port_available(port),
        )
    control = _median(observations[TwoAdapterPerformanceMode.GLOBAL_REFERENCE.value])
    left = _median(observations[ConventionalVariantBranch.LEFT.value])
    right = _median(observations[ConventionalVariantBranch.RIGHT.value])
    floor = left + right
    result = artifacts.root / "sdxl-conventional-variant-floor.json"
    result.write_text(
        json.dumps(
            {
                "status": "completed",
                "repeats": repeats,
                "observations": observations,
                "two_variant_floor_ms": floor,
                "two_variant_floor_to_global_ratio": floor / control,
                "advance_threshold": 2.50,
                "advance_passed": floor / control <= 2.50,
                "system_stats": system_stats,
                "cleanup": {
                    "server": server_cleanup,
                    "model_links": links.cleaned,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _measure(
    client: LoopbackComfyClient,
    *,
    label: str,
    repeats: int,
    build: WorkflowBuilder,
) -> dict[str, object]:
    """Warm and measure one deterministic workflow builder."""

    if not label:
        raise ValueError("Conventional floor timing label must be nonempty.")
    warmup = _execute(client, build(f"{label}:warmup"))
    measured = tuple(
        _execute(client, build(f"{label}:repeat-{index + 1}"))
        for index in range(repeats)
    )
    return {
        "warmup": asdict(warmup),
        "measured": [asdict(value) for value in measured],
        "median_runtime_ms": median_runtime_ms(measured),
    }


def _global_workflow(
    run_id: str,
    case: SdxlVisualCase,
) -> BuiltTwoAdapterPerformanceWorkflow:
    """Build the ordinary two-LoRA global control."""

    return build_two_adapter_performance_workflow(
        run_id=run_id,
        checkpoint_name=CHECKPOINT_SELECTION,
        mask_names=("unused-left", "unused-right"),
        case=case,
        mode=TwoAdapterPerformanceMode.GLOBAL_REFERENCE,
    )


def _variant_workflow(
    run_id: str,
    case: SdxlVisualCase,
    branch: ConventionalVariantBranch,
) -> BuiltConventionalVariantWorkflow:
    """Build one ordinary permanently applied adapter variant."""

    return build_conventional_variant_workflow(
        run_id=run_id,
        checkpoint_name=CHECKPOINT_SELECTION,
        case=case,
        branch=branch,
    )


def _variant_builder(
    case: SdxlVisualCase,
    branch: ConventionalVariantBranch,
) -> WorkflowBuilder:
    """Bind one branch without an untyped loop-capturing lambda."""

    def build(run_id: str) -> BuiltConventionalVariantWorkflow:
        """Build one exact branch for a unique metrics identifier."""

        return _variant_workflow(run_id, case, branch)

    return build


def _execute(
    client: LoopbackComfyClient,
    workflow: TimingWorkflow,
) -> SdxlRegionalLoraTiming:
    """Submit one typed workflow and decode synchronized timing."""

    prompt_id = client.submit(workflow.prompt)
    history = client.wait_for_history(prompt_id, timeout=1200.0)
    return decode_timing(history, workflow.metrics_node_id)


def _median(observation: dict[str, object]) -> float:
    """Narrow one recorded median from the local artifact structure."""

    value = observation.get("median_runtime_ms")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("Conventional floor median must be numeric.")
    return float(value)
