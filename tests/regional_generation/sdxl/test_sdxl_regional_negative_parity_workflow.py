# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify ordered regional negatives in the candidate SDXL parity workflow."""

from __future__ import annotations

from dataclasses import replace

from tools.comfy_api import JsonObject
from tools.sdxl_attention_couple_parity.cases import parity_case
from tools.sdxl_attention_couple_parity.workflow import (
    ParityBackend,
    build_parity_workflow,
)


def test_candidate_packs_base_left_and_right_negative_conditioning() -> None:
    """Carry every authored negative region to the public candidate sampler."""

    case = replace(
        parity_case(),
        left_negative_g="left negative G",
        left_negative_l="left negative L",
        right_negative_g="right negative G",
        right_negative_l="right negative L",
    )
    candidate = build_parity_workflow(
        backend=ParityBackend.CANDIDATE,
        run_id="run",
        checkpoint_name=r"owned\checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=case,
    )
    reference = build_parity_workflow(
        backend=ParityBackend.REFERENCE,
        run_id="run",
        checkpoint_name=r"owned\checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=case,
    )

    encodes = _inputs(candidate.prompt, "CLIPTextEncodeSDXL")
    assert len(encodes) == 6
    assert [encodes[index]["text_g"] for index in (3, 4, 5)] == [
        case.base_negative_g,
        "left negative G",
        "right negative G",
    ]
    assert [encodes[index]["text_l"] for index in (3, 4, 5)] == [
        case.base_negative_l,
        "left negative L",
        "right negative L",
    ]
    candidate_appends = _inputs(candidate.prompt, "SimpleSyrup.ConditioningBatchAppend")
    assert len(candidate_appends) == 4
    candidate_sampler = _sole(
        candidate.prompt,
        "SimpleSyrup.KSamplerAttentionCoupling",
    )
    assert candidate_sampler["negative"] == [
        _node_id(candidate.prompt, candidate_appends[-1]),
        0,
    ]
    assert _inputs(reference.prompt, "SimpleSyrup.ConditioningBatchAppend") == []
    reference_sampler = _sole(reference.prompt, "KSampler")
    assert reference_sampler["negative"] != candidate_sampler["negative"]


def _inputs(prompt: dict[str, JsonObject], class_type: str) -> list[JsonObject]:
    """Return typed inputs for every node of one class."""

    return [
        node["inputs"]
        for node in prompt.values()
        if node["class_type"] == class_type and isinstance(node["inputs"], dict)
    ]


def _sole(prompt: dict[str, JsonObject], class_type: str) -> JsonObject:
    """Return the sole input object for one class."""

    values = _inputs(prompt, class_type)
    assert len(values) == 1
    return values[0]


def _node_id(prompt: dict[str, JsonObject], inputs: JsonObject) -> str:
    """Return the graph node id owning one exact input object."""

    matches = [node_id for node_id, node in prompt.items() if node["inputs"] is inputs]
    assert len(matches) == 1
    return matches[0]
