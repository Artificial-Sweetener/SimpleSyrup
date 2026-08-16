# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lock generic native, attention, and regional-LoRA visual graphs."""

from __future__ import annotations

from typing import cast

from tools.anima_visual_performance_comparison.cases import visual_cases
from tools.anima_visual_performance_comparison.inventory import AnimaVisualInventory
from tools.anima_visual_performance_comparison.workflow import (
    build_anima_visual_workflow,
)


def test_anima_visual_graphs_change_only_the_execution_topology() -> None:
    """Require native, regional-only, then one full-strength regional adapter."""

    inventory = AnimaVisualInventory(*tuple(f"value-{index}" for index in range(10)))
    workflows = tuple(
        build_anima_visual_workflow(
            run_id="visual-run",
            case=case,
            inventory=inventory,
            mask_names=("left.png", "right.png"),
        )
        for case in visual_cases()
    )
    types = tuple(
        tuple(str(node["class_type"]) for node in workflow.prompt.values())
        for workflow in workflows
    )

    assert all(
        "SimpleSyrupBenchmark.StaticAnimaRegionalProfile" not in item for item in types
    )
    assert types[0].count("KSampler") == 1
    assert types[1].count("SimpleSyrup.KSamplerAttentionCoupling") == 1
    assert types[2].count("SimpleSyrup.KSamplerAttentionCoupling") == 1
    assert all("SimpleSyrupBenchmark.InstrumentModel" in item for item in types)
    regional_node = next(
        node
        for node in workflows[2].prompt.values()
        if node["class_type"] == "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl"
    )
    regional_inputs = cast(dict[str, object], regional_node["inputs"])
    assert "<lora:value-3:1>" in str(regional_inputs["positive_prompt"])
