# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the ordinary native SDXL global-LoRA reference graphs."""

from __future__ import annotations

from tools.comfy_api import JsonObject
from tools.sdxl_attention_couple_parity.cases import parity_case
from tools.sdxl_global_lora_reference.cases import NativeGlobalLoraCase
from tools.sdxl_global_lora_reference.workflow import (
    GLOBAL_LORA_STRENGTH,
    BuiltNativeGlobalLoraWorkflow,
    build_native_global_lora_workflow,
)


def test_native_pair_changes_only_the_ordinary_global_lora_load() -> None:
    """Keep prompt and sampler controls exact while adding MODEL plus CLIP LoRA."""

    control = _build(NativeGlobalLoraCase.TRIGGER_CONTROL)
    candidate = _build(NativeGlobalLoraCase.GLOBAL_LORA)

    control_encodes = _inputs(control.prompt, "CLIPTextEncodeSDXL")
    candidate_encodes = _inputs(candidate.prompt, "CLIPTextEncodeSDXL")
    assert len(control_encodes) == len(candidate_encodes) == 2
    for control_encode, candidate_encode in zip(
        control_encodes,
        candidate_encodes,
        strict=True,
    ):
        for key in (
            "width",
            "height",
            "crop_w",
            "crop_h",
            "target_width",
            "target_height",
            "text_g",
            "text_l",
        ):
            assert control_encode[key] == candidate_encode[key]
        assert control_encode["target_width"] == 1536
        assert control_encode["target_height"] == 1536
    positive_g = control_encodes[0]["text_g"]
    positive_l = control_encodes[0]["text_l"]
    assert isinstance(positive_g, str)
    assert isinstance(positive_l, str)
    assert positive_g.endswith(", style global trigger")
    assert positive_l.endswith(", style local trigger")
    assert "style global trigger" not in str(control_encodes[1]["text_g"])
    assert "style local trigger" not in str(control_encodes[1]["text_l"])

    control_sampler = _sole(control.prompt, "KSampler")
    candidate_sampler = _sole(candidate.prompt, "KSampler")
    for key in (
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "denoise",
    ):
        assert control_sampler[key] == candidate_sampler[key]
    assert _inputs(control.prompt, "EmptyLatentImage") == _inputs(
        candidate.prompt,
        "EmptyLatentImage",
    )
    assert not _inputs(control.prompt, "LoraLoader")
    loader = _sole(candidate.prompt, "LoraLoader")
    assert loader["lora_name"] == r"owned\style.safetensors"
    assert loader["strength_model"] == GLOBAL_LORA_STRENGTH
    assert loader["strength_clip"] == GLOBAL_LORA_STRENGTH


def test_native_reference_graph_uses_no_regional_or_custom_node() -> None:
    """Keep the causal reference entirely on installed native Comfy nodes."""

    for case in NativeGlobalLoraCase:
        workflow = _build(case)
        assert "KSampler" in workflow.required_node_ids
        assert all(
            not node_id.startswith("SimpleSyrup")
            for node_id in workflow.required_node_ids
        )
        assert not any(
            "AttentionCouple" in node_id for node_id in workflow.required_node_ids
        )


def _build(case: NativeGlobalLoraCase) -> BuiltNativeGlobalLoraWorkflow:
    """Build one anonymous deterministic graph."""

    return build_native_global_lora_workflow(
        case=case,
        run_id="run",
        checkpoint_name=r"owned\checkpoint.safetensors",
        lora_name=r"owned\style.safetensors",
        prompt_case=parity_case(),
        style_prompt_g="style global trigger",
        style_prompt_l="style local trigger",
    )


def _inputs(prompt: dict[str, JsonObject], class_type: str) -> list[JsonObject]:
    """Return every narrowed node-input object for one class."""

    values = [
        node["inputs"] for node in prompt.values() if node["class_type"] == class_type
    ]
    assert all(isinstance(value, dict) for value in values)
    return [value for value in values if isinstance(value, dict)]


def _sole(prompt: dict[str, JsonObject], class_type: str) -> JsonObject:
    """Return the sole node-input object for one class."""

    values = _inputs(prompt, class_type)
    assert len(values) == 1
    return values[0]
