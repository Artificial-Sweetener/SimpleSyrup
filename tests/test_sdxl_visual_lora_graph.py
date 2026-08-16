# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify focused full-fidelity SDXL visual-LoRA graph construction."""

from __future__ import annotations

from typing import cast

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.graph import SdxlWorkflowGraph
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    GlobalVisualAdapter,
    RegionalVisualAdapter,
)
from tools.sdxl_attention_coupling_integration.visual_lora_graph import (
    SdxlVisualLoraGraphBuilder,
)


def test_global_adapters_chain_model_and_clip_at_equal_authored_strength() -> None:
    """Apply each ordinary SDXL LoRA to both diffusion and text encoders."""

    graph = SdxlWorkflowGraph()
    loaded = SdxlVisualLoraGraphBuilder().load_global(
        graph,
        model=["model", 0],
        clip=["clip", 0],
        adapters=(
            GlobalVisualAdapter("first.safetensors", 0.25),
            GlobalVisualAdapter("second.safetensors", 0.75),
        ),
    )

    nodes = list(graph.prompt.values())
    assert [node["class_type"] for node in nodes] == ["LoraLoader", "LoraLoader"]
    assert _inputs(nodes[0])["strength_model"] == 0.25
    assert _inputs(nodes[0])["strength_clip"] == 0.25
    assert _inputs(nodes[1])["model"] == ["1", 0]
    assert _inputs(nodes[1])["clip"] == ["1", 1]
    assert loaded.model == ["2", 0]
    assert loaded.clip == ["2", 1]


def test_regional_hook_preserves_independent_model_and_clip_strengths() -> None:
    """Pass both authored regional strengths directly to Comfy's full hook."""

    graph = SdxlWorkflowGraph()
    hooks, identities = SdxlVisualLoraGraphBuilder().regional_hooks(
        graph,
        (RegionalVisualAdapter("character.safetensors", 0.6, 0.25),),
    )

    node = next(iter(graph.prompt.values()))
    assert node["class_type"] == "CreateHookLora"
    assert _inputs(node)["strength_model"] == 0.6
    assert _inputs(node)["strength_clip"] == 0.25
    assert hooks == ["1", 0]
    assert identities == ("character.safetensors",)


def _inputs(node: JsonObject) -> JsonObject:
    """Narrow one generated node input mapping."""

    value = node.get("inputs")
    if not isinstance(value, dict):
        raise AssertionError("Generated node inputs must be an object.")
    return cast(JsonObject, value)
