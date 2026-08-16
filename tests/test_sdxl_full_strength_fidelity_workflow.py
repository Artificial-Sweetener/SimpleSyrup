# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify fair global and regional full-strength fidelity workflows."""

from __future__ import annotations

from typing import cast

from tools.comfy_api import JsonObject
from tools.sdxl_full_strength_lora_fidelity.cases import (
    CharacterFidelitySpec,
    FidelityExecutionMode,
    fidelity_cases,
)
from tools.sdxl_full_strength_lora_fidelity.workflow import (
    build_fidelity_workflow,
)


def test_workflows_change_only_global_versus_all_one_regional_execution() -> None:
    """Keep prompts and sampling exact while changing LoRA placement semantics."""

    reference, regional = fidelity_cases(
        (
            CharacterFidelitySpec(
                "character",
                "Character",
                "character.safetensors",
                "trait",
                "trait",
            ),
        )
    )
    workflows = tuple(
        build_fidelity_workflow(
            run_id="proof",
            checkpoint_name="checkpoint.safetensors",
            mask_name="all-one.png",
            base_positive_g="1girl",
            base_positive_l="1girl",
            negative_g="bad quality",
            negative_l="bad quality",
            case=case,
        )
        for case in (reference, regional)
    )

    reference_nodes = tuple(workflows[0].prompt.values())
    regional_nodes = tuple(workflows[1].prompt.values())
    global_lora = _single(reference_nodes, "LoraLoader")
    regional_lora = _single(regional_nodes, "CreateHookLora")
    assert _inputs(global_lora)["strength_model"] == 1.0
    assert _inputs(global_lora)["strength_clip"] == 1.0
    assert _inputs(regional_lora)["strength_model"] == 1.0
    assert _inputs(regional_lora)["strength_clip"] == 1.0
    assert not any(
        node["class_type"] == "CreateHookKeyframe" for node in regional_nodes
    )
    assert _single(reference_nodes, "KSampler")["class_type"] == "KSampler"
    sampler = _single(regional_nodes, "SimpleSyrup.KSamplerAttentionCoupling")
    assert _inputs(sampler)["regional_prompt_weight"] == 1.0
    assert workflows[1].diagnostics_node_id is None
    assert not any(
        node["class_type"]
        in {
            "SimpleSyrupBenchmark.CaptureRegionalDiagnostics",
            "SimpleSyrupBenchmark.ReadRegionalDiagnostics",
        }
        for node in regional_nodes
    )
    assert reference.mode is FidelityExecutionMode.GLOBAL_REFERENCE
    assert regional.mode is FidelityExecutionMode.REGIONAL_ALL_ONE


def _single(nodes: tuple[JsonObject, ...], class_type: str) -> JsonObject:
    """Return one exact graph node with the requested class type."""

    matching = tuple(node for node in nodes if node.get("class_type") == class_type)
    assert len(matching) == 1
    return matching[0]


def _inputs(node: JsonObject) -> JsonObject:
    """Return one graph node's narrowed inputs."""

    value = node.get("inputs")
    assert isinstance(value, dict)
    return cast(JsonObject, value)
