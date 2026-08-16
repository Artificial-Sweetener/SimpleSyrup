# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize the focused SDXL visual sampling controls."""

from __future__ import annotations

from typing import cast

from tools.comfy_api import JsonObject
from tools.sdxl_attention_couple_parity.cases import parity_case
from tools.sdxl_attention_couple_parity.workflow import (
    ParityBackend,
    build_parity_workflow,
)


def test_candidate_workflow_uses_characterized_sampling_controls() -> None:
    """Preserve the exact base sampler controls through ownership extraction."""

    workflow = build_parity_workflow(
        backend=ParityBackend.CANDIDATE,
        run_id="sampling-control-characterization",
        checkpoint_name="checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=parity_case(),
    )

    sampler = next(
        node
        for node in workflow.prompt.values()
        if node["class_type"] == "SimpleSyrup.KSamplerAttentionCoupling"
    )
    inputs = cast(JsonObject, sampler["inputs"])

    assert inputs["seed"] == 7_429_113_057
    assert inputs["cfg"] == 5.0
    assert inputs["sampler_name"] == "euler_ancestral"
    assert inputs["scheduler"] == "karras"
    assert inputs["steps"] == 30
