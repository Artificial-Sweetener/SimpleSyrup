# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify matched vanilla and two-region SDXL performance workflows."""

from __future__ import annotations

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    RegionalVisualAdapter,
    SdxlVisualCase,
)
from tools.sdxl_regional_lora_performance.two_adapter_workflow import (
    TwoAdapterPerformanceMode,
    build_two_adapter_performance_workflow,
    build_two_adapter_steady_state_workflow,
)


def test_two_adapter_workflows_change_only_placement_and_sampler_topology() -> None:
    """Keep sampling locked while comparing global and regional placement."""

    built = tuple(
        build_two_adapter_performance_workflow(
            run_id=f"proof-{mode.value}",
            checkpoint_name="checkpoint.safetensors",
            mask_names=("left.png", "right.png"),
            case=_case(),
            mode=mode,
        )
        for mode in TwoAdapterPerformanceMode
    )
    global_nodes = tuple(built[0].prompt.values())
    regional_nodes = tuple(built[1].prompt.values())

    assert sum(node["class_type"] == "LoraLoader" for node in global_nodes) == 2
    assert sum(node["class_type"] == "CreateHookLora" for node in regional_nodes) == 2
    assert sum(node["class_type"] == "KSampler" for node in global_nodes) == 1
    assert (
        sum(
            node["class_type"] == "SimpleSyrup.KSamplerAttentionCoupling"
            for node in regional_nodes
        )
        == 1
    )
    class_types = tuple(
        node["class_type"] for graph in built for node in graph.prompt.values()
    )
    assert "VAEDecode" not in class_types
    assert "SaveImage" not in class_types
    assert not any(
        node["class_type"] == "SimpleSyrupBenchmark.CaptureRegionalDiagnostics"
        for graph in built
        for node in graph.prompt.values()
    )


def test_steady_state_workflow_keeps_prepared_model_stable_across_seeds() -> None:
    """Change only KSampler seed without installing a MODEL-cloning probe."""

    built = build_two_adapter_steady_state_workflow(
        checkpoint_name="checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=_case(),
        mode=TwoAdapterPerformanceMode.REGIONAL,
    )
    first = built.prompt_for_seed(101)
    second = built.prompt_for_seed(202)

    first_inputs = _node_inputs(first, built.sampler_node_id)
    second_inputs = _node_inputs(second, built.sampler_node_id)
    assert first_inputs["seed"] == 101
    assert second_inputs["seed"] == 202
    first_inputs["seed"] = 202
    assert first == second
    class_types = tuple(node["class_type"] for node in built.prompt.values())
    assert "SimpleSyrupBenchmark.InstrumentModel" not in class_types
    assert "SimpleSyrupBenchmark.ReadMetrics" not in class_types
    assert "SimpleSyrupBenchmark.CompleteLatent" in class_types


def test_steady_state_workflow_rejects_invalid_seed() -> None:
    """Fail closed before submitting a malformed seed-only revision."""

    built = build_two_adapter_steady_state_workflow(
        checkpoint_name="checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=_case(),
        mode=TwoAdapterPerformanceMode.GLOBAL_REFERENCE,
    )

    try:
        built.prompt_for_seed(-1)
    except ValueError as error:
        assert "non-negative integer" in str(error)
    else:
        raise AssertionError("Negative steady-state seed was accepted.")


def _case() -> SdxlVisualCase:
    """Return one generic full-strength adapter per side."""

    return SdxlVisualCase(
        "two-adapter",
        "Two adapter",
        base_positive_g="two subjects",
        base_positive_l="two subjects",
        left_g="left traits",
        left_l="left traits",
        right_g="right traits",
        right_l="right traits",
        left_adapters=(RegionalVisualAdapter("left.safetensors", 1.0, 1.0),),
        right_adapters=(RegionalVisualAdapter("right.safetensors", 1.0, 1.0),),
        regional_prompt_weight=1.0,
    )


def _node_inputs(prompt: dict[str, JsonObject], node_id: str) -> dict[str, object]:
    """Narrow one API node's dynamic input mapping for assertions."""

    inputs = prompt[node_id].get("inputs")
    if not isinstance(inputs, dict):
        raise TypeError("Test workflow node inputs must be an object.")
    return inputs
