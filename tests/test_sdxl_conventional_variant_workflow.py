# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the generic conventional single-variant timing graph."""

from __future__ import annotations

from collections.abc import Mapping

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    RegionalVisualAdapter,
    SdxlVisualCase,
)
from tools.sdxl_regional_lora_performance.conventional_variant_workflow import (
    ConventionalVariantBranch,
    build_conventional_variant_steady_state_workflow,
    build_conventional_variant_workflow,
)


def test_conventional_variant_uses_one_native_lora_and_plain_sampler() -> None:
    """Keep one adapter permanent with no regional sampler or HookGroup nodes."""

    workflow = build_conventional_variant_workflow(
        run_id="generic-run",
        checkpoint_name="generic-checkpoint.safetensors",
        case=_case(),
        branch=ConventionalVariantBranch.LEFT,
    )
    nodes = tuple(workflow.prompt.values())
    loaders = tuple(node for node in nodes if node["class_type"] == "LoraLoader")
    samplers = tuple(node for node in nodes if node["class_type"] == "KSampler")

    assert len(loaders) == 1
    loader_inputs = loaders[0].get("inputs")
    sampler_inputs = samplers[0].get("inputs") if samplers else None
    assert isinstance(loader_inputs, dict)
    assert isinstance(sampler_inputs, dict)
    assert loader_inputs["lora_name"] == "generic-left.safetensors"
    assert len(samplers) == 1
    assert sampler_inputs["steps"] == 30
    assert sampler_inputs["sampler_name"] == "euler_ancestral"
    assert not any(
        str(node["class_type"]).startswith("SimpleSyrup.KSampler") for node in nodes
    )
    assert not any("Hook" in str(node["class_type"]) for node in nodes)


def test_right_variant_selects_only_the_right_generic_adapter() -> None:
    """Prove branch selection without inventory-identity routing."""

    workflow = build_conventional_variant_workflow(
        run_id="generic-run",
        checkpoint_name="generic-checkpoint.safetensors",
        case=_case(),
        branch=ConventionalVariantBranch.RIGHT,
    )
    loader = next(
        node for node in workflow.prompt.values() if node["class_type"] == "LoraLoader"
    )
    loader_inputs = loader.get("inputs")

    assert isinstance(loader_inputs, dict)
    assert loader_inputs["lora_name"] == "generic-right.safetensors"


def test_conventional_variant_steady_state_changes_only_sampler_seed() -> None:
    """Keep one prepared native LoRA graph stable across measured submissions."""

    built = build_conventional_variant_steady_state_workflow(
        checkpoint_name="generic-checkpoint.safetensors",
        case=_case(),
        branch=ConventionalVariantBranch.LEFT,
    )
    first = built.prompt_for_seed(101)
    second = built.prompt_for_seed(202)

    first_sampler = _node_inputs(first, built.sampler_node_id)
    second_sampler = _node_inputs(second, built.sampler_node_id)
    assert first_sampler["seed"] == 101
    assert second_sampler["seed"] == 202
    first_sampler["seed"] = 202
    assert first == second
    class_types = tuple(node["class_type"] for node in built.prompt.values())
    assert class_types.count("LoraLoader") == 1
    assert class_types.count("KSampler") == 1
    assert "SimpleSyrupBenchmark.CompleteLatent" in class_types
    assert "SimpleSyrupBenchmark.InstrumentModel" not in class_types
    assert "SimpleSyrupBenchmark.ReadMetrics" not in class_types


def _case() -> SdxlVisualCase:
    """Return a model-neutral two-branch fixture with no local inventory names."""

    return SdxlVisualCase(
        case_id="generic",
        label="Generic",
        base_positive_g="base g",
        base_positive_l="base l",
        base_negative_g="negative g",
        base_negative_l="negative l",
        left_g="left g",
        left_l="left l",
        right_g="right g",
        right_l="right l",
        left_adapters=(RegionalVisualAdapter("generic-left.safetensors", 1.0, 1.0),),
        right_adapters=(RegionalVisualAdapter("generic-right.safetensors", 1.0, 1.0),),
    )


def _node_inputs(
    prompt: Mapping[str, JsonObject],
    node_id: str,
) -> dict[str, object]:
    """Narrow one generated node input mapping for seed assertions."""

    node = prompt[node_id]
    if not isinstance(node, dict):
        raise TypeError("Test workflow node must be an object.")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise TypeError("Test workflow node inputs must be an object.")
    return inputs
