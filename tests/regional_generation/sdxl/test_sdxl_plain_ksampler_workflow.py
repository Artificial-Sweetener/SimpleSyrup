# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove the plain SDXL comparator contains no regional or LoRA execution."""

from __future__ import annotations

from typing import cast

from tools.comfy_api import JsonObject
from tools.sdxl_full_strength_lora_fidelity.plain_workflow import (
    build_plain_ksampler_workflow,
)


def test_plain_ksampler_workflow_is_native_and_uses_locked_controls() -> None:
    """Require one ordinary KSampler without LoRA, SEP, mask, or regional nodes."""

    workflow = build_plain_ksampler_workflow(
        run_id="plain-run",
        checkpoint_name="checkpoint.safetensors",
        positive_g="pink-haired subject",
        positive_l="pink-haired subject",
        negative_g="bad quality",
        negative_l="bad quality",
    )
    nodes = tuple(workflow.prompt.values())
    types = tuple(str(node["class_type"]) for node in nodes)
    sampler = next(node for node in nodes if node["class_type"] == "KSampler")
    inputs = cast(JsonObject, sampler["inputs"])

    assert types.count("KSampler") == 1
    assert not any("Lora" in node_type for node_type in types)
    assert not any("AttentionCoupling" in node_type for node_type in types)
    assert not any("Mask" in node_type for node_type in types)
    assert inputs["seed"] == 7429113057
    assert inputs["steps"] == 30
    assert inputs["cfg"] == 5.0
    assert inputs["sampler_name"] == "euler_ancestral"
    assert inputs["scheduler"] == "karras"
