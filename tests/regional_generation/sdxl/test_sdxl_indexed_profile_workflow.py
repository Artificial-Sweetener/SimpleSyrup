# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify generic indexed SDXL operator-profile graph insertion."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools.comfy_api import JsonObject
from tools.sdxl_regional_lora_performance.indexed_profile_workflow import (
    build_indexed_operator_profile_workflow,
    decode_indexed_operator_profile,
)


def test_profile_insertion_copies_graph_and_wraps_current_sampler_model() -> None:
    """Preserve the source prompt while rewiring one selected sampler."""

    prompt = _prompt()
    original = copy.deepcopy(prompt)
    built = build_indexed_operator_profile_workflow(
        prompt=prompt,
        sampler_node_id="sampler",
        run_id="generic-profile",
        trace_path=Path("E:/artifacts/trace.json"),
        call_index=30,
    )

    assert prompt == original
    sampler_inputs = _inputs(built.prompt, "sampler")
    profile_inputs = _inputs(built.prompt, built.profile_model_node_id)
    result_inputs = _inputs(built.prompt, built.profile_node_id)
    assert profile_inputs["model"] == ["source-model", 0]
    assert profile_inputs["call_index"] == 30
    assert sampler_inputs["model"] == [built.profile_model_node_id, 0]
    assert result_inputs["latent"] == ["sampler", 0]
    assert "SimpleSyrupBenchmark.ProfileIndexedModelCall" in built.required_node_ids
    assert "SimpleSyrupBenchmark.ReadOperatorProfile" in built.required_node_ids


def test_profile_decoder_returns_one_complete_capture() -> None:
    """Narrow the benchmark UI payload at one authoritative boundary."""

    capture = decode_indexed_operator_profile(
        {
            "outputs": {
                "profile-result": {
                    "operator_profile": [{"call_index": 30, "operators": []}]
                }
            }
        },
        "profile-result",
    )

    assert capture == {"call_index": 30, "operators": []}


def test_profile_decoder_rejects_missing_terminal() -> None:
    """Fail closed before publishing an incomplete diagnostic artifact."""

    with pytest.raises(ValueError, match="missing its terminal"):
        decode_indexed_operator_profile({"outputs": {}}, "profile-result")


def _prompt() -> dict[str, JsonObject]:
    """Return one minimal sampler graph with an existing model reference."""

    return {
        "source-model": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "sampler": {
            "class_type": "SimpleSyrup.KSamplerAttentionCoupling",
            "inputs": {"model": ["source-model", 0], "seed": 1},
        },
    }


def _inputs(prompt: dict[str, JsonObject], node_id: str) -> dict[str, object]:
    """Narrow one generated node input mapping."""

    inputs = prompt[node_id].get("inputs")
    if not isinstance(inputs, dict):
        raise TypeError("Profile test node inputs must be an object.")
    return inputs
