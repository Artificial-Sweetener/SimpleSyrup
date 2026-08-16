# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify native SDXL regional-LoRA conditioning graph ownership."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from sdxl_visual_test_inventory import visual_inventory

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.graph import SdxlWorkflowGraph
from tools.sdxl_attention_coupling_integration.visual_cases import visual_cases
from tools.sdxl_attention_coupling_integration.visual_conditioning import (
    SdxlVisualConditioningBuilder,
)
from tools.sdxl_attention_coupling_integration.visual_prompt_defaults import (
    BASE_NEGATIVE_L,
    BASE_POSITIVE_G,
    BASE_POSITIVE_L,
)


def test_global_style_and_regional_character_use_distinct_comfy_paths(
    tmp_path: Path,
) -> None:
    """Keep global style on MODEL and character hooks on regional conditioning."""

    prompt = _build("global-style-regional-character", tmp_path)
    global_loaders = _nodes(prompt, "LoraLoader")
    hooks = _nodes(prompt, "CreateHookLora")
    attached = _nodes(prompt, "ConditioningSetProperties")

    assert len(global_loaders) == 1
    assert len(hooks) == 1
    assert _inputs(global_loaders[0])["lora_name"] != _inputs(hooks[0])["lora_name"]
    assert _inputs(global_loaders[0])["strength_clip"] == 0.65
    assert _inputs(hooks[0])["strength_clip"] == 0.45
    assert len(attached) == 2
    assert len(_nodes(prompt, "CLIPTextEncodeSDXL")) == 6
    assert "SimpleSyrup.AttachRegionalGlobalConditioning" not in {
        node["class_type"] for node in prompt.values()
    }
    positive = _positive_encodes(prompt)
    inventory = visual_inventory(tmp_path)
    assert inventory.style.prompt_g in str(_inputs(positive[0])["text_g"])
    assert inventory.style.prompt_l in str(_inputs(positive[0])["text_l"])
    assert inventory.right_character.prompt_g not in str(_inputs(positive[0])["text_g"])
    assert inventory.right_character.prompt_l not in str(_inputs(positive[0])["text_l"])
    assert inventory.style.prompt_g in str(_inputs(positive[2])["text_g"])
    assert inventory.style.prompt_l in str(_inputs(positive[2])["text_l"])
    assert inventory.right_character.prompt_g in str(_inputs(positive[2])["text_g"])
    assert inventory.right_character.prompt_l in str(_inputs(positive[2])["text_l"])


def test_base_encoder_stays_global_while_each_region_owns_its_character_prompt(
    tmp_path: Path,
) -> None:
    """Keep character details out of base and exclusive to their owning region."""

    prompt = _build("different-characters-prompt-control", tmp_path)
    positive = _positive_encodes(prompt)
    inventory = visual_inventory(tmp_path)
    base_inputs = _inputs(positive[0])
    left_inputs = _inputs(positive[1])
    right_inputs = _inputs(positive[2])

    assert base_inputs["text_g"] == BASE_POSITIVE_G
    assert base_inputs["text_l"] == BASE_POSITIVE_L
    assert inventory.left_character.prompt_g in str(left_inputs["text_g"])
    assert inventory.left_character.prompt_l in str(left_inputs["text_l"])
    assert inventory.right_character.prompt_g not in str(left_inputs["text_g"])
    assert inventory.right_character.prompt_l not in str(left_inputs["text_l"])
    assert inventory.right_character.prompt_g in str(right_inputs["text_g"])
    assert inventory.right_character.prompt_l in str(right_inputs["text_l"])
    assert inventory.left_character.prompt_g not in str(right_inputs["text_g"])
    assert inventory.left_character.prompt_l not in str(right_inputs["text_l"])


def test_regional_model_hook_is_paired_across_positive_and_negative_cfg(
    tmp_path: Path,
) -> None:
    """Preserve the right region slot while pairing its model/CLIP variant."""

    prompt = _build("global-style-regional-character", tmp_path)
    attached = _nodes(prompt, "ConditioningSetProperties")
    encoded = _nodes(prompt, "CLIPTextEncodeSDXL")
    negative = [node for node in encoded if _inputs(node)["text_l"] == BASE_NEGATIVE_L]

    assert len(_nodes(prompt, "SimpleSyrup.ConditioningBatchStart")) == 2
    assert len(_nodes(prompt, "SimpleSyrup.ConditioningBatchAppend")) == 4
    assert len(attached) == 2
    assert len(negative) == 3
    assert _inputs(attached[0])["hooks"] == _inputs(attached[1])["hooks"]


def test_layout_first_case_schedules_only_regional_positive_contexts(
    tmp_path: Path,
) -> None:
    """Leave both base branches active while delaying two regional positives."""

    prompt = _build("global-style-layout-first-control", tmp_path)
    scheduled = _nodes(prompt, "ConditioningSetTimestepRange")

    assert len(scheduled) == 2
    assert [_inputs(node)["start"] for node in scheduled] == [0.15, 0.15]
    assert [_inputs(node)["end"] for node in scheduled] == [1.0, 1.0]
    assert len(_nodes(prompt, "CLIPTextEncodeSDXL")) == 4


def test_same_adapter_in_both_regions_retains_two_authored_hook_uses(
    tmp_path: Path,
) -> None:
    """Do not collapse the same adapter identity across regional owners."""

    prompt = _build("same-style-both", tmp_path)
    hooks = _nodes(prompt, "CreateHookLora")
    labels = _nodes(prompt, "SimpleSyrup.LabelRegionalLoraHooks")

    assert len(hooks) == 2
    assert _inputs(hooks[0])["lora_name"] == _inputs(hooks[1])["lora_name"]
    assert len(labels) == 2
    identities = [
        json.loads(cast(str, _inputs(label)["adapter_identities_json"]))
        for label in labels
    ]
    assert identities[0] == identities[1]


def test_same_adapter_global_and_regional_retains_both_authored_uses(
    tmp_path: Path,
) -> None:
    """Keep ordinary global strength and left regional strength independent."""

    prompt = _build("same-style-global-left", tmp_path)
    loader = _nodes(prompt, "LoraLoader")[0]
    hook = _nodes(prompt, "CreateHookLora")[0]

    assert _inputs(loader)["lora_name"] == _inputs(hook)["lora_name"]
    assert _inputs(loader)["strength_model"] == 0.35
    assert _inputs(loader)["strength_clip"] == 0.35
    assert _inputs(hook)["strength_model"] == 0.55
    assert _inputs(hook)["strength_clip"] == 0.55
    inventory = visual_inventory(tmp_path)
    positive = _positive_encodes(prompt)
    assert [
        str(_inputs(node)["text_g"]).count(inventory.style.prompt_g)
        for node in positive
    ] == [1, 2, 1]
    assert [
        str(_inputs(node)["text_l"]).count(inventory.style.prompt_l)
        for node in positive
    ] == [1, 2, 1]


def test_regional_encoders_combine_shared_global_with_only_their_section(
    tmp_path: Path,
) -> None:
    """Encode shared composition through each region's G and L conditioning."""

    case = next(
        item
        for item in visual_cases(visual_inventory(tmp_path))
        if item.case_id == "baseline"
    )
    positive = _positive_encodes(_build(case.case_id, tmp_path))

    assert _inputs(positive[0])["text_g"] == BASE_POSITIVE_G
    assert _inputs(positive[0])["text_l"] == BASE_POSITIVE_L
    assert _inputs(positive[1])["text_g"] == f"{BASE_POSITIVE_G}, {case.left_g}"
    assert _inputs(positive[1])["text_l"] == f"{BASE_POSITIVE_L}, {case.left_l}"
    assert _inputs(positive[2])["text_g"] == f"{BASE_POSITIVE_G}, {case.right_g}"
    assert _inputs(positive[2])["text_l"] == f"{BASE_POSITIVE_L}, {case.right_l}"


def test_multi_adapter_order_and_schedule_are_explicit(tmp_path: Path) -> None:
    """Preserve declared adapter order and independent Comfy keyframes."""

    multiple = _build("multiple-left", tmp_path)
    hooks = _nodes(multiple, "CreateHookLora")
    label = _nodes(multiple, "SimpleSyrup.LabelRegionalLoraHooks")[0]
    assert [_inputs(hook)["lora_name"] for hook in hooks] == json.loads(
        cast(str, _inputs(label)["adapter_identities_json"])
    )
    assert len(_nodes(multiple, "CombineHooks2")) == 1

    scheduled = _build("scheduled-right-character", tmp_path)
    keyframes = _nodes(scheduled, "CreateHookKeyframe")
    assert [_inputs(node)["start_percent"] for node in keyframes] == [0.0, 0.6]
    assert [_inputs(node)["strength_mult"] for node in keyframes] == [1.0, 0.0]
    assert len(_nodes(scheduled, "SetHookKeyframes")) == 1


def _build(case_id: str, tmp_path: Path) -> dict[str, JsonObject]:
    """Build one case against stable test MODEL and CLIP references."""

    case = next(
        item
        for item in visual_cases(visual_inventory(tmp_path))
        if item.case_id == case_id
    )
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


def _positive_encodes(prompt: dict[str, JsonObject]) -> list[JsonObject]:
    """Return the three authored positive encodes in graph order."""

    return [
        node
        for node in _nodes(prompt, "CLIPTextEncodeSDXL")
        if _inputs(node)["text_l"] != BASE_NEGATIVE_L
    ]
