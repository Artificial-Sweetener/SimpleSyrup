# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact public graph construction for every P9.1 case."""

from __future__ import annotations

from typing import cast

from tools.comfy_api import JsonObject
from tools.prompt_control_attention_coupling_integration.matrix import (
    PUBLIC_NODE_ID,
    STEPS,
    cases,
)
from tools.prompt_control_attention_coupling_integration.workflow import (
    PromptControlAttentionWorkflowBuilder,
)


def test_every_workflow_uses_public_pc_hooks_batches_and_one_sampler() -> None:
    """Keep regional LoRAs off the supplied model and inside conditioning only."""

    builder = PromptControlAttentionWorkflowBuilder()
    for case in cases():
        workflow = builder.build(
            case,
            run_id="run",
            mask_names=("left.png", "right.png"),
        )
        types = _types(workflow.prompt)
        sampler = _sole(workflow.prompt, PUBLIC_NODE_ID)
        latent = _sole(workflow.prompt, "EmptyCosmosLatentVideo")

        assert types >= {
            "SimpleSyrupBenchmark.SnapshotPromptControlExpansion",
            "SimpleSyrupBenchmark.SnapshotPromptControl",
            "SimpleSyrup.ConditioningBatchStart",
            "SimpleSyrup.ConditioningBatchAppend",
            "SimpleSyrupBenchmark.InstrumentModel",
            "SimpleSyrupBenchmark.CaptureRegionalDiagnostics",
            PUBLIC_NODE_ID,
        }
        assert "LoraLoaderModelOnly" not in types
        assert _inputs(sampler)["steps"] == STEPS
        assert _inputs(sampler)["cfg"] == 4.0
        assert _inputs(latent)["width"] == 512
        assert _inputs(latent)["height"] == 512
        assert workflow.conditioning_evidence.expansion_node_id is not None
        assert workflow.conditioning_evidence.snapshot_node_id is not None
        if case.has_regional_hooks:
            assert types >= {
                "PCLazyLoraLoaderAdvanced",
                "SimpleSyrup.LabelRegionalLoraHooks",
                "SimpleSyrup.PrepareRegionalLoraHooks",
                "ConditioningSetProperties",
                "SimpleSyrup.AttachRegionalGlobalConditioning",
            }
            schedulers = _nodes(workflow.prompt, "PCLazyLoraLoaderAdvanced")
            assert len(schedulers) == 1
            assert "model" not in _inputs(schedulers[0])
            assert "clip" not in _inputs(schedulers[0])
        else:
            assert "PCLazyLoraLoaderAdvanced" not in types


def test_overlap_and_inactive_cases_share_exact_p0_8_text_builder() -> None:
    """Keep explicit overlap and zero-width shapes out of P9 graph policy."""

    by_id = {case.case_id: case for case in cases()}
    builder = PromptControlAttentionWorkflowBuilder()
    overlap = builder.build(
        by_id["text-overlapping"],
        run_id="run",
        mask_names=("left.png", "right.png"),
    ).prompt
    inactive = builder.build(
        by_id["text-inactive"],
        run_id="run",
        mask_names=("left.png", "right.png"),
    ).prompt

    overlap_ranges = _nodes(overlap, "PCTextEncodeWithRange")
    assert any(
        _inputs(node).get("start") == 0.0 and _inputs(node).get("end") == 0.75
        for node in overlap_ranges
    )
    assert any(
        _inputs(node).get("start") == 0.25 and _inputs(node).get("end") == 1.0
        for node in overlap_ranges
    )
    assert len(_nodes(overlap, "ConditioningCombine")) == 1
    assert any(
        _inputs(node).get("start") == 0.5 and _inputs(node).get("end") == 0.5
        for node in _nodes(inactive, "PCTextEncodeWithRange")
    )


def test_regional_negatives_use_exact_characterized_p0_8_intervals() -> None:
    """Carry immutable negative ranges without interpreting LoRA schedules."""

    builder = PromptControlAttentionWorkflowBuilder()
    for case in cases():
        workflow = builder.build(
            case,
            run_id="run",
            mask_names=("left.png", "right.png"),
        )
        regional_negatives = tuple(
            node
            for node in _nodes(workflow.prompt, "PCTextEncodeWithRange")
            if _inputs(node).get("text") == "low quality"
        )

        assert tuple(
            (_inputs(node).get("start"), _inputs(node).get("end"))
            for node in regional_negatives
        ) == tuple(
            (expected.start, expected.end)
            for expected in case.characterization.expected_negative
        )


def _types(prompt: dict[str, JsonObject]) -> set[str]:
    """Return graph class types."""

    return {str(node["class_type"]) for node in prompt.values()}


def _nodes(prompt: dict[str, JsonObject], class_type: str) -> list[JsonObject]:
    """Return all nodes of one class."""

    return [node for node in prompt.values() if node["class_type"] == class_type]


def _sole(prompt: dict[str, JsonObject], class_type: str) -> JsonObject:
    """Return the sole node of one class."""

    nodes = _nodes(prompt, class_type)
    assert len(nodes) == 1
    return nodes[0]


def _inputs(node: JsonObject) -> dict[str, object]:
    """Narrow one node input mapping."""

    return cast(dict[str, object], node["inputs"])
