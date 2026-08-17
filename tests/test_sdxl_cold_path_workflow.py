# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the focused SDXL cold-attribution graph."""

from __future__ import annotations

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    RegionalVisualAdapter,
    SdxlVisualCase,
)
from tools.sdxl_regional_lora_performance.cold_path_workflow import (
    build_sdxl_cold_path_workflow,
)


def test_cold_workflow_is_image_free_and_revises_only_seed_and_capture() -> None:
    """Keep the measured preparation graph stable across cold and warm runs."""

    built = build_sdxl_cold_path_workflow(
        checkpoint_name="checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=_case(),
    )

    first = built.prompt_for_execution(seed=101, run_id="cold")
    second = built.prompt_for_execution(seed=202, run_id="warm")
    assert _inputs(first, built.sampler_node_id)["seed"] == 101
    assert _inputs(second, built.sampler_node_id)["seed"] == 202
    assert _inputs(first, built.capture_node_id)["run_id"] == "cold"
    assert _inputs(first, built.terminal_node_id)["run_id"] == "cold"
    _inputs(first, built.sampler_node_id)["seed"] = 202
    _inputs(first, built.capture_node_id)["run_id"] = "warm"
    _inputs(first, built.terminal_node_id)["run_id"] = "warm"
    assert first == second
    class_types = tuple(node["class_type"] for node in built.prompt.values())
    assert class_types.count("SimpleSyrup.KSamplerAttentionCoupling") == 1
    assert class_types.count("SimpleSyrupBenchmark.CaptureColdPathDiagnostics") == 1
    assert class_types.count("SimpleSyrupBenchmark.ReadColdPathDiagnostics") == 1
    assert "VAEDecode" not in class_types
    assert "SaveImage" not in class_types
    assert "SimpleSyrupBenchmark.InstrumentModel" not in class_types


def _case() -> SdxlVisualCase:
    """Return two anonymous full-strength character adapters."""

    return SdxlVisualCase(
        "cold-path",
        "Cold path",
        base_positive_g="two subjects",
        base_positive_l="two subjects",
        left_g="left traits",
        left_l="left traits",
        right_g="right traits",
        right_l="right traits",
        left_adapters=(RegionalVisualAdapter("left.safetensors", 1.0, 1.0),),
        right_adapters=(RegionalVisualAdapter("right.safetensors", 1.0, 1.0),),
        regional_prompt_weight=1.0,
    )


def _inputs(prompt: dict[str, JsonObject], node_id: str) -> dict[str, object]:
    """Narrow one test graph input mapping."""

    inputs = prompt[node_id].get("inputs")
    if not isinstance(inputs, dict):
        raise TypeError("Cold-path test node inputs must be an object.")
    return inputs
