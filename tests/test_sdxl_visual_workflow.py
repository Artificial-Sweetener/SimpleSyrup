# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify one-case-at-a-time U11 managed SDXL workflow graphs."""

from __future__ import annotations

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.visual_cases import visual_cases
from tools.sdxl_attention_coupling_integration.visual_workflow import (
    build_sdxl_visual_workflow,
)


def test_baseline_builds_only_one_native_full_sampler() -> None:
    """Keep ordinary baseline work to one source trajectory and six G/L encodes."""

    case = next(item for item in visual_cases() if item.case_id == "baseline")
    built = build_sdxl_visual_workflow(
        run_id="run",
        checkpoint_name=r"owned\checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=case,
    )
    classes = _class_counts(built.prompt)

    assert [output.artifact_id for output in built.outputs] == ["baseline--full"]
    assert classes["SimpleSyrup.KSamplerAttentionCoupling"] == 1
    assert "SimpleSyrup.KSamplerAttentionCouplingTiled" not in classes
    assert "SimpleSyrup.KSamplerAttentionCouplingContextual" not in classes
    assert classes["CLIPTextEncodeSDXL"] == 6
    assert "CLIPTextEncode" not in classes
    assert "CreateHookLoraModelOnly" not in classes
    assert "LoraLoaderModelOnly" not in classes


def test_selected_case_builds_full_tiled_and_contextual_from_one_source() -> None:
    """Refine only the accepted global-style plus regional-character source."""

    case = next(
        item
        for item in visual_cases()
        if item.case_id == "global-style-regional-character"
    )
    built = build_sdxl_visual_workflow(
        run_id="run",
        checkpoint_name=r"owned\checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=case,
    )
    classes = _class_counts(built.prompt)

    assert [output.artifact_id for output in built.outputs] == [
        "global-style-regional-character--full",
        "global-style-regional-character--tiled-1.5x",
        "global-style-regional-character--contextual-1.5x",
    ]
    assert classes["SimpleSyrup.KSamplerAttentionCoupling"] == 1
    assert classes["SimpleSyrup.KSamplerAttentionCouplingTiled"] == 1
    assert classes["SimpleSyrup.KSamplerAttentionCouplingContextual"] == 1
    assert classes["ImageScale"] == 1
    assert classes["VAEEncode"] == 1
    assert classes["VAEDecode"] == 3
    assert classes["LoraLoaderModelOnly"] == 1
    assert classes["CreateHookLoraModelOnly"] == 1


def _class_counts(prompt: dict[str, JsonObject]) -> dict[str, int]:
    """Count exact API node classes for concise graph assertions."""

    counts: dict[str, int] = {}
    for node in prompt.values():
        class_type = node["class_type"]
        if not isinstance(class_type, str):
            raise AssertionError("Generated class_type must be text.")
        counts[class_type] = counts.get(class_type, 0) + 1
    return counts
