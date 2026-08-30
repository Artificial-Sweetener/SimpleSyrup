# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test prompt-level attention-region planning and graph rewriting."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from simple_syrup.domain.attention_region_capture import AttentionRegionRequestKind
from simple_syrup.domain.attention_spatial_transform import (
    AttentionSpatialTransformKind,
)
from simple_syrup.runtime.attention_region_graph import (
    ATTENTION_REGION_PROMPT_PLANNER,
    ATTENTION_REGION_PROMPT_REWRITER,
    INTERNAL_CAPTURE_NODE_ID,
)
from simple_syrup.runtime.attention_region_plan_codec import (
    ATTENTION_CAPTURE_PLAN_CODEC,
)


class _ImagePass:
    """Declare an exact IMAGE passthrough for provenance tests."""

    GRAPH_PASSTHROUGH_OUTPUTS = {0: "image"}


class _LatentPass:
    """Declare an exact LATENT passthrough for provenance tests."""

    GRAPH_PASSTHROUGH_OUTPUTS = {0: "latent"}


def test_image_requests_coalesce_and_rewrite_one_direct_sampler_model_edge() -> None:
    """Plan two downstream requests as one capture on their shared KSampler."""

    prompt = _direct_prompt()
    plans = ATTENTION_REGION_PROMPT_PLANNER.build(prompt, {"ImagePass": _ImagePass})

    assert len(plans) == 1
    plan = plans[0]
    assert plan.sampler_node_id == "sampler"
    assert plan.model_owner_node_id == "sampler"
    assert plan.model_link == ("loader", 0)
    assert plan.positive_link == ("positive", 0)
    assert plan.prompt_text == "1girl, pink hair, twintails, smug, selfie"
    assert plan.clip_link == ("loader", 1)
    assert tuple(request.kind for request in plan.requests) == (
        AttentionRegionRequestKind.ALL_PROMPT_SEGS,
        AttentionRegionRequestKind.CONCEPT_SEGS,
    )

    ATTENTION_REGION_PROMPT_REWRITER.rewrite(prompt, plans)

    capture_id = plan.capture_node_id
    assert prompt["sampler"]["inputs"]["model"] == [capture_id, 0]
    capture = prompt[capture_id]
    assert capture["class_type"] == INTERNAL_CAPTURE_NODE_ID
    assert capture["inputs"]["model"] == ["loader", 0]
    assert capture["inputs"]["positive"] == ["positive", 0]
    assert capture["inputs"]["clip"] == ["loader", 1]
    serialized = json.loads(capture["inputs"]["plan_json"])
    assert [request["node_id"] for request in serialized["requests"]] == [
        "all",
        "search",
    ]


def test_latent_request_traces_through_declared_passthrough_to_advanced_guider() -> (
    None
):
    """Resolve the MODEL owner behind SamplerCustomAdvanced before execution."""

    prompt: dict[str, Any] = {
        "model": {"class_type": "LoadModel", "inputs": {}},
        "positive": {"class_type": "Conditioning", "inputs": {}},
        "guider": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["model", 0],
                "positive": ["positive", 0],
                "negative": ["negative", 0],
            },
        },
        "sampler": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "guider": ["guider", 0],
                "latent_image": ["empty", 0],
            },
        },
        "empty": {"class_type": "EmptyLatentImage", "inputs": {}},
        "pass": {"class_type": "LatentPass", "inputs": {"latent": ["sampler", 1]}},
        "mask": _request_node(
            "SimpleSyrup.AttentionRegionMask", "latent", ["pass", 0], "head"
        ),
    }

    plans = ATTENTION_REGION_PROMPT_PLANNER.build(prompt, {"LatentPass": _LatentPass})

    assert len(plans) == 1
    plan = plans[0]
    assert plan.sampler_node_id == "sampler"
    assert plan.model_owner_node_id == "guider"
    ATTENTION_REGION_PROMPT_REWRITER.rewrite(prompt, plans)
    assert prompt["guider"]["inputs"]["model"] == [plan.capture_node_id, 0]


def test_unsupported_provenance_is_observable_and_not_rewritten(caplog: Any) -> None:
    """Leave opaque image paths untouched while logging the exact request id."""

    prompt: dict[str, Any] = {
        "loaded": {"class_type": "LoadImage", "inputs": {}},
        "search": _request_node(
            "SimpleSyrup.ConceptAttentionSEGS", "image", ["loaded", 0], "head"
        ),
    }

    plans = ATTENTION_REGION_PROMPT_PLANNER.build(prompt, {})
    ATTENTION_REGION_PROMPT_REWRITER.rewrite(prompt, plans)

    assert plans == ()
    assert set(prompt) == {"loaded", "search"}
    assert "cannot recover a sampler" in caplog.text
    assert caplog.records[0].request_node_id == "search"


def test_post_render_controls_do_not_invalidate_sampler_capture_cache() -> None:
    """Keep upstream capture identity stable while users reshape saved maps."""

    original = _direct_prompt()
    changed_shape = deepcopy(original)
    shape_inputs = changed_shape["search"]["inputs"]
    shape_inputs.update(
        {
            "minimum_strength": 0.91,
            "minimum_consensus": 0.82,
            "geometry_recall": 0.24,
            "split_sensitivity": 0.73,
            "instance_recall": 0.91,
            "minimum_region_size": 99,
            "keep_only": 2,
            "keep_by": "highest confidence",
            "combine_segs": True,
            "matte_solidity": 1.0,
            "edge_feather": 24,
        }
    )
    changed_capture = deepcopy(original)
    changed_capture["search"]["inputs"]["capture_start"] = 0.42

    original_plan = ATTENTION_REGION_PROMPT_PLANNER.build(
        original, {"ImagePass": _ImagePass}
    )
    shape_plan = ATTENTION_REGION_PROMPT_PLANNER.build(
        changed_shape, {"ImagePass": _ImagePass}
    )
    capture_plan = ATTENTION_REGION_PROMPT_PLANNER.build(
        changed_capture, {"ImagePass": _ImagePass}
    )
    ATTENTION_REGION_PROMPT_REWRITER.rewrite(original, original_plan)
    ATTENTION_REGION_PROMPT_REWRITER.rewrite(changed_shape, shape_plan)
    ATTENTION_REGION_PROMPT_REWRITER.rewrite(changed_capture, capture_plan)

    original_json = original[original_plan[0].capture_node_id]["inputs"]["plan_json"]
    shape_json = changed_shape[shape_plan[0].capture_node_id]["inputs"]["plan_json"]
    capture_json = changed_capture[capture_plan[0].capture_node_id]["inputs"][
        "plan_json"
    ]
    assert shape_json == original_json
    assert capture_json != original_json


def _direct_prompt() -> dict[str, Any]:
    """Return one decoded-image graph containing two attention requests."""

    prompt: dict[str, Any] = {
        "loader": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["loader", 1],
                "text": "1girl, pink hair, twintails, smug, selfie",
            },
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["loader", 0],
                "positive": ["positive", 0],
                "latent_image": ["empty", 0],
            },
        },
        "empty": {"class_type": "EmptyLatentImage", "inputs": {}},
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sampler", 0], "vae": ["loader", 2]},
        },
        "image_pass": {
            "class_type": "ImagePass",
            "inputs": {"image": ["decode", 0]},
        },
        "search": _request_node(
            "SimpleSyrup.ConceptAttentionSEGS",
            "image",
            ["image_pass", 0],
            "pink hair | twintails",
        ),
        "all": _request_node(
            "SimpleSyrup.AllPromptAttentionSEGS",
            "image",
            ["decode", 0],
            "",
        ),
    }
    return prompt


def _request_node(
    class_type: str,
    source_name: str,
    source_link: list[object],
    queries: str,
) -> dict[str, Any]:
    """Return one serialized public request with representative controls."""

    return {
        "class_type": class_type,
        "inputs": {
            source_name: source_link,
            "concepts": queries,
            "sampler_stage": 1,
            "capture_start": 0.1,
            "capture_end": 0.8,
            "minimum_strength": 0.4,
            "minimum_consensus": 0.3,
            "split_sensitivity": 0.6,
            "minimum_region_size": 12,
            "capture_profile": "balanced",
        },
    }


def test_sampler_stage_selects_first_direct_or_clamped_lineage_stage() -> None:
    """Resolve one-based sampler stages through decode, resize, and encode."""

    prompt = _two_stage_prompt()
    first = ATTENTION_REGION_PROMPT_PLANNER.build(prompt, {})
    assert tuple(plan.sampler_node_id for plan in first) == ("first",)
    assert tuple(
        transform.kind for transform in first[0].requests[0].spatial_transforms
    ) == (AttentionSpatialTransformKind.SCALE,)
    ATTENTION_REGION_PROMPT_REWRITER.rewrite(prompt, first)
    serialized = prompt[first[0].capture_node_id]["inputs"]["plan_json"]
    decoded = ATTENTION_CAPTURE_PLAN_CODEC.decode(serialized)
    assert decoded.requests[0].spatial_transforms == (
        first[0].requests[0].spatial_transforms
    )
    assert decoded.source_aspect == 768 / 896

    prompt["concept"]["inputs"]["sampler_stage"] = 0
    direct = ATTENTION_REGION_PROMPT_PLANNER.build(prompt, {})
    assert tuple(plan.sampler_node_id for plan in direct) == ("second",)

    prompt["concept"]["inputs"]["sampler_stage"] = 99
    clamped = ATTENTION_REGION_PROMPT_PLANNER.build(prompt, {})
    assert tuple(plan.sampler_node_id for plan in clamped) == ("second",)


def test_generic_latent_sampler_inputs_are_compatible_without_class_allowlist() -> None:
    """Patch a custom latent sampler that follows Comfy's standard input contract."""

    prompt = _direct_prompt()
    prompt["sampler"]["class_type"] = "ThirdPartySampler"

    plans = ATTENTION_REGION_PROMPT_PLANNER.build(prompt, {"ImagePass": _ImagePass})

    assert tuple(plan.sampler_node_id for plan in plans) == ("sampler",)


def test_crop_local_detailer_remains_in_lineage_but_selected_capture_is_no_op(
    caplog: Any,
) -> None:
    """Allow earlier-stage selection through a detailer without misregistered maps."""

    prompt = _detailer_chain_prompt()
    first = ATTENTION_REGION_PROMPT_PLANNER.build(prompt, {})
    assert tuple(plan.sampler_node_id for plan in first) == ("sampler",)

    prompt["search"]["inputs"]["sampler_stage"] = 2
    crop_local = ATTENTION_REGION_PROMPT_PLANNER.build(prompt, {})

    assert crop_local == ()
    assert "crop-local images" in caplog.records[-1].reason


def _two_stage_prompt() -> dict[str, Any]:
    """Return a full-canvas hires lineage with two explicit samplers."""

    return {
        "loader": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["loader", 1], "text": "girl, pink hair"},
        },
        "empty": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 768, "height": 896, "batch_size": 1},
        },
        "first": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["loader", 0],
                "positive": ["positive", 0],
                "latent_image": ["empty", 0],
            },
        },
        "decode_first": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["first", 0], "vae": ["loader", 2]},
        },
        "scale": {
            "class_type": "ImageScaleBy",
            "inputs": {"image": ["decode_first", 0], "scale_by": 2.0},
        },
        "encode": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["scale", 0], "vae": ["loader", 2]},
        },
        "second": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["loader", 0],
                "positive": ["positive", 0],
                "latent_image": ["encode", 0],
            },
        },
        "decode_second": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["second", 0], "vae": ["loader", 2]},
        },
        "concept": _request_node(
            "SimpleSyrup.ConceptAttentionSEGS",
            "image",
            ["decode_second", 0],
            "pink hair",
        ),
    }


def _detailer_chain_prompt() -> dict[str, Any]:
    """Return one ordinary sampler followed by a crop-local detailer stage."""

    prompt = _direct_prompt()
    prompt["detailer"] = {
        "class_type": "SimpleSyrup.DetailSEGSByScaleFactor",
        "inputs": {
            "image": ["decode", 0],
            "model": ["loader", 0],
            "positive": ["positive", 0],
        },
    }
    prompt["search"]["inputs"]["image"] = ["detailer", 0]
    del prompt["all"]
    return prompt
