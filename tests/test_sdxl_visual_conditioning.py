# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify native SDXL regional-LoRA conditioning graph ownership."""

from __future__ import annotations

import json
from typing import cast

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.graph import SdxlWorkflowGraph
from tools.sdxl_attention_coupling_integration.visual_cases import visual_cases
from tools.sdxl_attention_coupling_integration.visual_conditioning import (
    SdxlVisualConditioningBuilder,
)


def test_global_style_and_regional_character_use_distinct_comfy_paths() -> None:
    """Keep global style on MODEL and CHARACTER_B on the right conditioning only."""

    prompt = _build("global-style-regional-character")
    global_loaders = _nodes(prompt, "LoraLoaderModelOnly")
    hooks = _nodes(prompt, "CreateHookLoraModelOnly")
    attached = _nodes(prompt, "ConditioningSetProperties")

    assert len(global_loaders) == 1
    assert len(hooks) == 1
    assert _inputs(global_loaders[0])["lora_name"] != _inputs(hooks[0])["lora_name"]
    assert len(attached) == 2
    assert len(_nodes(prompt, "CLIPTextEncodeSDXL")) == 6
    assert "SimpleSyrup.AttachRegionalGlobalConditioning" not in {
        node["class_type"] for node in prompt.values()
    }


def test_same_adapter_in_both_regions_retains_two_authored_hook_uses() -> None:
    """Do not collapse the same adapter identity across regional owners."""

    prompt = _build("same-style-both")
    hooks = _nodes(prompt, "CreateHookLoraModelOnly")
    labels = _nodes(prompt, "SimpleSyrup.LabelRegionalLoraHooks")

    assert len(hooks) == 2
    assert _inputs(hooks[0])["lora_name"] == _inputs(hooks[1])["lora_name"]
    assert len(labels) == 2
    identities = [
        json.loads(cast(str, _inputs(label)["adapter_identities_json"]))
        for label in labels
    ]
    assert identities[0] == identities[1]


def test_same_adapter_global_and_regional_retains_both_authored_uses() -> None:
    """Keep ordinary global strength and left regional strength independent."""

    prompt = _build("same-style-global-left")
    loader = _nodes(prompt, "LoraLoaderModelOnly")[0]
    hook = _nodes(prompt, "CreateHookLoraModelOnly")[0]

    assert _inputs(loader)["lora_name"] == _inputs(hook)["lora_name"]
    assert _inputs(loader)["strength_model"] == 0.35
    assert _inputs(hook)["strength_model"] == 0.55


def test_multi_adapter_order_and_schedule_are_explicit() -> None:
    """Preserve declared adapter order and independent Comfy keyframes."""

    multiple = _build("multiple-left")
    hooks = _nodes(multiple, "CreateHookLoraModelOnly")
    label = _nodes(multiple, "SimpleSyrup.LabelRegionalLoraHooks")[0]
    assert [_inputs(hook)["lora_name"] for hook in hooks] == json.loads(
        cast(str, _inputs(label)["adapter_identities_json"])
    )
    assert len(_nodes(multiple, "CombineHooks2")) == 1

    scheduled = _build("scheduled-right-character")
    keyframes = _nodes(scheduled, "CreateHookKeyframe")
    assert [_inputs(node)["start_percent"] for node in keyframes] == [0.0, 0.6]
    assert [_inputs(node)["strength_mult"] for node in keyframes] == [1.0, 0.0]
    assert len(_nodes(scheduled, "SetHookKeyframes")) == 1


def _build(case_id: str) -> dict[str, JsonObject]:
    """Build one case against stable test MODEL and CLIP references."""

    case = next(item for item in visual_cases() if item.case_id == case_id)
    graph = SdxlWorkflowGraph()
    SdxlVisualConditioningBuilder().build(
        graph,
        case=case,
        model=["model", 0],
        clip=["clip", 0],
    )
    return graph.prompt


def _nodes(prompt: dict[str, JsonObject], class_type: str) -> list[JsonObject]:
    """Return graph nodes with one exact class type."""

    return [node for node in prompt.values() if node["class_type"] == class_type]


def _inputs(node: JsonObject) -> JsonObject:
    """Narrow one generated node's input mapping."""

    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise AssertionError("Generated node inputs must be an object.")
    return cast(JsonObject, inputs)
