# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the public full and 1.5x SDXL managed workflow graph."""

from __future__ import annotations

from typing import cast

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.matrix import (
    CONTEXTUAL_EXPECTED_MODEL_CALLS,
    MODES,
    NEGATIVE_PROMPTS,
    NEGATIVE_PROMPTS_G,
    POSITIVE_PROMPTS,
    POSITIVE_PROMPTS_G,
    REFINEMENT_DENOISE,
    REFINEMENT_STEPS,
    SOURCE_STEPS,
    TILE_BATCH_SIZE,
    TILE_OVERLAP,
    TILE_SIZE,
)
from tools.sdxl_attention_coupling_integration.workflow import (
    build_sdxl_attention_coupling_workflow,
)


def test_workflow_uses_three_public_nodes_and_exact_upscale_product_flow() -> None:
    """Keep full 1024 generation before tiled and Contextual 1536 refinement."""

    workflow = build_sdxl_attention_coupling_workflow(
        run_id="run-1",
        checkpoint_name="owned\\checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
    )
    by_class = _nodes_by_class(workflow.prompt)

    assert set(workflow.outputs) == {mode.mode_id for mode in MODES}
    assert len(by_class["SimpleSyrup.KSamplerAttentionCoupling"]) == 1
    assert len(by_class["SimpleSyrup.KSamplerAttentionCouplingTiled"]) == 1
    assert len(by_class["SimpleSyrup.KSamplerAttentionCouplingContextual"]) == 1
    empty_inputs = _inputs(by_class["EmptyLatentImage"][0])
    assert empty_inputs["width"] == 1024
    assert empty_inputs["height"] == 1024
    assert "LatentUpscaleBy" not in by_class
    assert len(by_class["VAEDecode"]) == 3
    assert len(by_class["ImageScale"]) == 1
    assert len(by_class["VAEEncode"]) == 1
    upscale_inputs = _inputs(by_class["ImageScale"][0])
    assert upscale_inputs["upscale_method"] == "lanczos"
    assert upscale_inputs["width"] == 1536
    assert upscale_inputs["height"] == 1536
    assert upscale_inputs["crop"] == "disabled"
    source_decode = upscale_inputs["image"]
    assert isinstance(source_decode, list)
    assert workflow.prompt[str(source_decode[0])]["class_type"] == "VAEDecode"
    encoded_node_id = next(
        node_id
        for node_id, node in workflow.prompt.items()
        if node["class_type"] == "VAEEncode"
    )
    tiled_inputs = _inputs(by_class["SimpleSyrup.KSamplerAttentionCouplingTiled"][0])
    contextual_inputs = _inputs(
        by_class["SimpleSyrup.KSamplerAttentionCouplingContextual"][0]
    )
    assert tiled_inputs["steps"] == REFINEMENT_STEPS
    assert tiled_inputs["denoise"] == REFINEMENT_DENOISE
    assert tiled_inputs["latent_tile_width"] == TILE_SIZE
    assert tiled_inputs["latent_tile_overlap"] == TILE_OVERLAP
    assert tiled_inputs["latent_tile_batch_size"] == TILE_BATCH_SIZE
    assert tiled_inputs["latent_image"] == [encoded_node_id, 0]
    assert contextual_inputs["latent_context_size"] == TILE_SIZE
    assert contextual_inputs["latent_context_overlap"] == TILE_OVERLAP
    assert contextual_inputs["global_steps"] == 1
    assert contextual_inputs["latent_image"] == [encoded_node_id, 0]
    assert MODES[2].expected_model_calls == CONTEXTUAL_EXPECTED_MODEL_CALLS == 61
    full_inputs = _inputs(by_class["SimpleSyrup.KSamplerAttentionCoupling"][0])
    assert full_inputs["steps"] == SOURCE_STEPS
    assert "CLIPTextEncode" not in by_class
    clip_inputs = [_inputs(node) for node in by_class["CLIPTextEncodeSDXL"]]
    assert [inputs["text_l"] for inputs in clip_inputs] == [
        *POSITIVE_PROMPTS,
        *NEGATIVE_PROMPTS,
    ]
    assert [inputs["text_g"] for inputs in clip_inputs] == [
        *POSITIVE_PROMPTS_G,
        *NEGATIVE_PROMPTS_G,
    ]
    assert all(
        inputs["width"] == 1024
        and inputs["height"] == 1024
        and inputs["crop_w"] == 0
        and inputs["crop_h"] == 0
        and inputs["target_width"] == 1536
        and inputs["target_height"] == 1536
        for inputs in clip_inputs
    )
    assert len(by_class["SimpleSyrupBenchmark.InstrumentModel"]) == 3
    assert len(by_class["SimpleSyrupBenchmark.CaptureRegionalDiagnostics"]) == 3


def _nodes_by_class(
    prompt: dict[str, JsonObject],
) -> dict[str, list[JsonObject]]:
    """Index API nodes by exact class type for readable graph assertions."""

    result: dict[str, list[JsonObject]] = {}
    for node in prompt.values():
        class_type = node["class_type"]
        if not isinstance(class_type, str):
            raise AssertionError("workflow class_type must be a string")
        result.setdefault(class_type, []).append(node)
    return result


def _inputs(node: JsonObject) -> JsonObject:
    """Narrow one API node's input mapping."""

    value = node.get("inputs")
    if not isinstance(value, dict):
        raise AssertionError("workflow inputs must be an object")
    return cast(JsonObject, value)
