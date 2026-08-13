# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove the exact P9.6 matrix and shared public graph construction."""

from __future__ import annotations

import json

from tools.anima_regional_lora_admission_integration.graph_contract import (
    PUBLIC_NODE_ID,
    STEPS,
)
from tools.anima_regional_lora_admission_integration.matrix import cases as p95_cases
from tools.anima_regional_lora_admission_integration.workflow import (
    BuiltRegionalLoraAdmissionWorkflow,
    RegionalLoraAdmissionWorkflowBuilder,
)
from tools.anima_regional_nondiffusion_rejection_integration.matrix import (
    NON_DIFFUSION_OWNER_BUNDLE_FIXTURE,
    TURBO_LORA_NAME,
    AnimaNondiffusionRejectionCase,
    cases,
)
from tools.attention_coupling_benchmark.comfy_probe.regional_hook_fixture import (
    CreateRegionalHookFixtureV3,
)
from tools.comfy_api import JsonObject


def test_matrix_declares_the_four_exact_non_diffusion_rejections() -> None:
    """Keep real, isolated, aggregate, and mixed atomicity cases closed."""

    definitions = cases()

    assert [case.case_id for case in definitions] == [
        "turbo-mixed-diffusion-llm",
        "llm-adapter-only",
        "text-encoder-vae-bundle",
        "adapter_a-plus-nondiffusion-bundle",
    ]
    assert [case.expected_issue_count for case in definitions] == [60, 1, 2, 2]
    assert [case.expected_issue_adapter_index for case in definitions] == [0, 0, 0, 1]


def test_p95_and_p96_fixture_wires_exhaust_the_host_combo_contract() -> None:
    """Keep both client matrices aligned with the independent custom node."""

    advertised = set(CreateRegionalHookFixtureV3.define_schema().inputs[0].options)
    declared = {
        case.fixture for case in (*p95_cases(), *cases()) if case.fixture is not None
    }

    assert declared == advertised


def test_real_turbo_graph_uses_one_public_lora_and_one_public_sampler() -> None:
    """Route the exact public Turbo payload to regional admission unchanged."""

    built = _build(cases()[0])
    prompt = built.workflow.prompt
    classes = _classes(prompt)

    assert built.artifact_phase == "p9.6"
    assert classes.count("CreateHookLoraModelOnly") == 1
    assert "SimpleSyrupBenchmark.CreateRegionalHookFixture" not in classes
    assert classes.count(PUBLIC_NODE_ID) == 1
    public_lora = _only_node(prompt, "CreateHookLoraModelOnly")
    assert _inputs(public_lora) == {
        "lora_name": TURBO_LORA_NAME,
        "strength_model": 1.0,
    }
    sampler = prompt[built.sampler_node_id]
    assert _inputs(sampler)["steps"] == STEPS


def test_mixed_graph_preserves_adapter_a_then_nondiffusion_bundle_order() -> None:
    """Require adapter-one ownership without reinterpreting fixture payloads."""

    built = _build(cases()[-1])
    prompt = built.workflow.prompt
    label = _only_node(prompt, "SimpleSyrup.LabelRegionalLoraHooks")
    combined = _only_node(prompt, "CombineHooks2")
    fixture = _only_node(
        prompt,
        "SimpleSyrupBenchmark.CreateRegionalHookFixture",
    )

    assert json.loads(str(_inputs(label)["adapter_identities_json"])) == [
        "supported-adapter_a",
        NON_DIFFUSION_OWNER_BUNDLE_FIXTURE,
    ]
    assert _link(_inputs(combined)["hooks_A"])[0] == _node_id(
        prompt, "CreateHookLoraModelOnly"
    )
    assert _link(_inputs(combined)["hooks_B"])[0] == _node_id(
        prompt, "SimpleSyrupBenchmark.CreateRegionalHookFixture"
    )
    assert _inputs(fixture)["fixture"] == NON_DIFFUSION_OWNER_BUNDLE_FIXTURE


def _build(
    case: AnimaNondiffusionRejectionCase,
) -> BuiltRegionalLoraAdmissionWorkflow:
    """Build one exact synthetic managed graph."""

    return RegionalLoraAdmissionWorkflowBuilder(artifact_phase="p9.6").build(
        case,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )


def _classes(prompt: dict[str, JsonObject]) -> list[str]:
    """Return graph class names in stable node order."""

    return [str(node["class_type"]) for node in prompt.values()]


def _only_node(prompt: dict[str, JsonObject], class_type: str) -> JsonObject:
    """Return one uniquely typed graph node."""

    nodes = [node for node in prompt.values() if node["class_type"] == class_type]
    assert len(nodes) == 1
    return nodes[0]


def _node_id(prompt: dict[str, JsonObject], class_type: str) -> str:
    """Return one uniquely typed graph node identity."""

    identities = [
        node_id for node_id, node in prompt.items() if node["class_type"] == class_type
    ]
    assert len(identities) == 1
    return identities[0]


def _inputs(node: JsonObject) -> JsonObject:
    """Narrow one graph node's input mapping."""

    inputs = node.get("inputs")
    assert isinstance(inputs, dict)
    assert all(isinstance(key, str) for key in inputs)
    return inputs


def _link(value: object) -> list[str | int]:
    """Narrow one graph node-reference value."""

    assert isinstance(value, list)
    assert all(isinstance(item, str | int) for item in value)
    return value
