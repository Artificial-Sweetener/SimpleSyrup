# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the focused sampler-free materialization parity graph."""

from __future__ import annotations

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    RegionalVisualAdapter,
    SdxlVisualCase,
)
from tools.sdxl_materialization_parity.workflow import (
    build_materialization_parity_workflow,
)


def test_workflow_reuses_regional_topology_without_sampling_or_decode() -> None:
    """Keep the comparison graph image-free and preserve its semantic inputs."""

    case = SdxlVisualCase(
        "materialization-parity",
        "Materialization parity",
        base_positive_g="two subjects",
        base_positive_l="two subjects",
        left_g="left traits",
        left_l="left traits",
        right_g="right traits",
        right_l="right traits",
        left_adapters=(RegionalVisualAdapter("adapter-a", 1.0, 1.0),),
        right_adapters=(RegionalVisualAdapter("adapter-b", 1.0, 1.0),),
        region_mask_feather=7,
        regional_prompt_weight=1.0,
    )

    built = build_materialization_parity_workflow(
        run_id="comparison-1",
        checkpoint_name="checkpoint",
        mask_names=("mask-a", "mask-b"),
        case=case,
    )

    class_types = tuple(str(node["class_type"]) for node in built.prompt.values())
    assert class_types.count("SimpleSyrupBenchmark.CompareMaterializationParity") == 1
    assert not any("KSampler" in class_type for class_type in class_types)
    assert "VAEDecode" not in class_types
    assert "SaveImage" not in class_types
    inputs = _inputs(built.prompt, built.terminal_node_id)
    assert inputs["run_id"] == "comparison-1"
    assert inputs["region_mask_feather"] == 7
    assert inputs["positive"] != inputs["negative"]
    assert inputs["region_masks"] == inputs["region_masks"]


def _inputs(prompt: dict[str, JsonObject], node_id: str) -> dict[str, object]:
    """Narrow one test graph input mapping."""

    inputs = prompt[node_id].get("inputs")
    if not isinstance(inputs, dict):
        raise TypeError("Materialization parity node inputs must be an object.")
    return inputs
