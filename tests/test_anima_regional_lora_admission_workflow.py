# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove exact P9.5 managed matrix and public graph construction."""

from __future__ import annotations

import json

from tools.anima_regional_lora_admission_integration.graph_contract import (
    PUBLIC_NODE_ID,
    STEPS,
)
from tools.anima_regional_lora_admission_integration.matrix import (
    RegionalLoraAdmissionCase,
    cases,
)
from tools.anima_regional_lora_admission_integration.workflow import (
    BuiltRegionalLoraAdmissionWorkflow,
    RegionalLoraAdmissionWorkflowBuilder,
)
from tools.attention_coupling_benchmark.comfy_probe.regional_hook_fixture import (
    CreateRegionalHookFixtureV3,
)
from tools.comfy_api import JsonObject


def test_matrix_declares_one_success_and_six_exact_rejections() -> None:
    """Keep supported, format, target, hook, and mixed atomicity cases closed."""

    definitions = cases()

    assert len(definitions) == 7
    assert [case.case_id for case in definitions] == [
        "supported-adapter_a",
        "non-weight-hook",
        "model-as-lora-format",
        "unsupported-suffix-format",
        "incomplete-standard-pair",
        "unsupported-anima-target",
        "mixed-supported-unsupported-target",
    ]
    assert [case.expect_success for case in definitions] == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_supported_graph_uses_public_adapter_a_hook_and_one_public_sampler() -> None:
    """Route the real adapter through public Comfy and SimpleSyrup nodes."""

    built = _build(cases()[0])
    classes = _classes(built.workflow.prompt)

    assert classes.count("CreateHookLoraModelOnly") == 1
    assert classes.count("SimpleSyrup.LabelRegionalLoraHooks") == 1
    assert "SimpleSyrupBenchmark.CreateRegionalHookFixture" not in classes
    assert "CombineHooks2" not in classes
    assert classes.count(PUBLIC_NODE_ID) == 1
    sampler = built.workflow.prompt[built.sampler_node_id]
    assert _inputs(sampler)["steps"] == STEPS
    label = _only_node(built.workflow.prompt, "SimpleSyrup.LabelRegionalLoraHooks")
    assert json.loads(str(_inputs(label)["adapter_identities_json"])) == [
        "supported-adapter_a"
    ]


def test_non_weight_graph_skips_the_weight_identity_boundary() -> None:
    """Let the production selector classify an exact non-weight hook itself."""

    built = _build(cases()[1])
    classes = _classes(built.workflow.prompt)

    assert "SimpleSyrupBenchmark.CreateRegionalHookFixture" in classes
    assert "SimpleSyrup.LabelRegionalLoraHooks" not in classes
    fixture = _only_node(
        built.workflow.prompt,
        "SimpleSyrupBenchmark.CreateRegionalHookFixture",
    )
    assert _inputs(fixture)["fixture"] == "non-weight-hook"


def test_matrix_fixtures_match_the_probe_public_combo_contract() -> None:
    """Keep serialized client values aligned with the independent host package."""

    advertised = set(CreateRegionalHookFixtureV3.define_schema().inputs[0].options)
    declared = {case.fixture for case in cases() if case.fixture is not None}

    assert declared < advertised
    assert advertised - declared == {
        "llm-adapter-target",
        "non-diffusion-owner-bundle",
    }


def test_mixed_graph_preserves_supported_then_unsupported_weight_order() -> None:
    """Prove the atomicity case labels both ordered hooks without reclassification."""

    built = _build(cases()[-1])
    prompt = built.workflow.prompt

    assert _classes(prompt).count("CombineHooks2") == 1
    label = _only_node(prompt, "SimpleSyrup.LabelRegionalLoraHooks")
    assert json.loads(str(_inputs(label)["adapter_identities_json"])) == [
        "supported-adapter_a",
        "unsupported-anima-target",
    ]
    combined = _only_node(prompt, "CombineHooks2")
    assert _link(_inputs(combined)["hooks_A"])[0] == _node_id(
        prompt, "CreateHookLoraModelOnly"
    )
    assert _link(_inputs(combined)["hooks_B"])[0] == _node_id(
        prompt,
        "SimpleSyrupBenchmark.CreateRegionalHookFixture",
    )


def test_every_graph_attaches_hooks_only_to_the_first_regional_positive() -> None:
    """Keep global conditioning unhooked and isolate one model-admission source."""

    for case in cases():
        prompt = _build(case).workflow.prompt
        assert _classes(prompt).count("ConditioningSetProperties") == 1
        assert _classes(prompt).count("SimpleSyrup.ConditioningBatchStart") == 2
        assert _classes(prompt).count("SimpleSyrup.ConditioningBatchAppend") == 4


def _build(
    case: RegionalLoraAdmissionCase,
) -> BuiltRegionalLoraAdmissionWorkflow:
    """Build one case with stable synthetic managed identities."""

    return RegionalLoraAdmissionWorkflowBuilder(artifact_phase="p9.5").build(
        case,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )


def _classes(prompt: dict[str, JsonObject]) -> list[str]:
    """Return graph class names in stable node order."""

    return [str(node["class_type"]) for node in prompt.values()]


def _only_node(
    prompt: dict[str, JsonObject],
    class_type: str,
) -> JsonObject:
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
