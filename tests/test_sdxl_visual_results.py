# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify complete incremental U11 visual evidence persistence."""

from __future__ import annotations

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import cast

from PIL import Image
from sdxl_visual_test_inventory import visual_inventory

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.matrix import MODES, SdxlIntegrationMode
from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    STYLE_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    SdxlVisualCase,
    VisualMode,
)
from tools.sdxl_attention_coupling_integration.visual_cases import (
    visual_cases,
)
from tools.sdxl_attention_coupling_integration.visual_history import (
    decode_sdxl_visual_history,
)
from tools.sdxl_attention_coupling_integration.visual_results import (
    SdxlVisualResultRecorder,
)
from tools.sdxl_attention_coupling_integration.visual_runtime_expectations import (
    SDXL_VISUAL_RUNTIME_EXPECTATIONS,
)
from tools.sdxl_attention_coupling_integration.visual_workflow import (
    BuiltSdxlVisualWorkflow,
    build_sdxl_visual_workflow,
)


def test_recorder_serializes_prompts_from_the_exact_case(tmp_path: Path) -> None:
    """Keep external global and regional negatives in durable result evidence."""

    base = visual_cases(visual_inventory(tmp_path))[0]
    case = replace(
        base,
        base_positive_g="external base positive g",
        base_positive_l="external base positive l",
        base_negative_g="external base negative g",
        base_negative_l="external base negative l",
        left_negative_g="external left negative g",
        left_negative_l="external left negative l",
        right_negative_g="external right negative g",
        right_negative_l="external right negative l",
    )
    recorder = SdxlVisualResultRecorder(tmp_path, cases=(case,))
    workflow = build_sdxl_visual_workflow(
        run_id="external-prompts",
        checkpoint_name=r"owned\checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=case,
        seed=7_429_113_058,
    )
    history = _history(workflow, case)
    recorder.record_case(
        workflow,
        case=case,
        history=history,
        prompt_id="external-prompt",
        evidence=decode_sdxl_visual_history(history, workflow),
        image_bytes={workflow.outputs[0].artifact_id: _png(1024, 1024, 1)},
        wall_runtime_ms=100.0,
    )
    decoded = json.loads((tmp_path / "u11-result.json").read_text(encoding="utf-8"))
    prompts = decoded["observations"][0]["prompts"]

    assert decoded["observations"][0]["seed"] == 7_429_113_058
    assert prompts == {
        "base_positive_g": case.base_positive_g,
        "base_positive_l": case.base_positive_l,
        "base_negative_g": case.base_negative_g,
        "base_negative_l": case.base_negative_l,
        "left_positive_g": case.left_g,
        "left_positive_l": case.left_l,
        "left_negative_g": case.left_negative_g,
        "left_negative_l": case.left_negative_l,
        "right_positive_g": case.right_g,
        "right_positive_l": case.right_l,
        "right_negative_g": case.right_negative_g,
        "right_negative_l": case.right_negative_l,
    }


def test_recorder_requires_all_declared_outputs_and_exact_cleanup(
    tmp_path: Path,
) -> None:
    """Persist every case incrementally and complete only after cleanup proof."""

    cases = visual_cases(visual_inventory(tmp_path))
    recorder = SdxlVisualResultRecorder(tmp_path, cases=cases)
    for index, case in enumerate(cases, 1):
        workflow = build_sdxl_visual_workflow(
            run_id="run",
            checkpoint_name=r"owned\checkpoint.safetensors",
            mask_names=("left.png", "right.png"),
            case=case,
        )
        history = _history(workflow, case)
        evidence = decode_sdxl_visual_history(history, workflow)
        images = {
            output.artifact_id: _png(
                _mode(output.mode).width,
                _mode(output.mode).height,
                index,
            )
            for output in workflow.outputs
        }
        recorder.record_case(
            workflow,
            case=case,
            history=history,
            prompt_id=f"prompt-{index}",
            evidence=evidence,
            image_bytes=images,
            wall_runtime_ms=100.0,
        )

    result_path = recorder.finalize(
        system_stats={"system": "test"},
        server_cleanup=True,
        model_cleanup=True,
        mask_cleanup=True,
        mask_evidence=({"profile": "hard"},),
    )
    decoded: object = json.loads(result_path.read_text(encoding="utf-8"))
    assert isinstance(decoded, dict)
    result = cast(dict[str, object], decoded)
    observations = result["observations"]
    assert isinstance(observations, list)
    assert result["status"] == "completed"
    assert len(observations) == 30
    assert [item["artifact_id"] for item in observations if isinstance(item, dict)] == [
        f"{case.case_id}--{mode.value}" for case in cases for mode in case.modes
    ]
    first = observations[0]
    assert isinstance(first, dict)
    prompts = first["prompts"]
    assert isinstance(prompts, dict)
    assert prompts["left_positive_l"] == cases[0].left_l
    assert prompts["right_positive_g"] == cases[0].right_g
    assert first["regional_prompt_weight"] == 0.4
    full_regional = next(
        item
        for item in observations
        if isinstance(item, dict)
        and item.get("case_id") == "global-style-full-regional-control"
    )
    assert full_regional["regional_prompt_weight"] == 1.0
    prompt_control = next(
        item
        for item in observations
        if isinstance(item, dict)
        and item.get("case_id") == "global-style-right-character-prompt-control"
    )
    regional_character = next(
        item
        for item in observations
        if isinstance(item, dict)
        and item.get("case_id") == "global-style-regional-character"
    )
    assert prompt_control["regional_prompt_weight"] == 1.0
    assert regional_character["regional_prompt_weight"] == 1.0
    model_only = next(
        item
        for item in observations
        if isinstance(item, dict)
        and item.get("case_id") == "multiple-left-model-only-style"
    )
    assert model_only["left_adapters"] == [
        {
            "name": STYLE_SELECTION,
            "model_strength": 0.55,
            "clip_strength": 0.0,
            "schedule": [[0.0, 1.0]],
        }
    ]


def test_failure_is_retained_without_erasing_prior_case(tmp_path: Path) -> None:
    """Keep earlier accepted evidence when a later managed case fails."""

    cases = visual_cases(visual_inventory(tmp_path))
    recorder = SdxlVisualResultRecorder(tmp_path, cases=cases)
    first, failed = cases[:2]
    workflow = build_sdxl_visual_workflow(
        run_id="run",
        checkpoint_name=r"owned\checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=first,
    )
    history = _history(workflow, first)
    evidence = decode_sdxl_visual_history(history, workflow)
    recorder.record_case(
        workflow,
        case=first,
        history=history,
        prompt_id="prompt-1",
        evidence=evidence,
        image_bytes={workflow.outputs[0].artifact_id: _png(1024, 1024, 1)},
        wall_runtime_ms=100.0,
    )
    recorder.record_failure(failed, RuntimeError("retained failure"))

    decoded: object = json.loads(
        (tmp_path / "u11-result.json").read_text(encoding="utf-8")
    )
    assert isinstance(decoded, dict)
    result = cast(dict[str, object], decoded)
    assert result["status"] == "failed"
    assert len(cast(list[object], result["observations"])) == 1
    assert cast(list[dict[str, object]], result["failures"])[0]["case_id"] == (
        failed.case_id
    )


def _history(
    workflow: BuiltSdxlVisualWorkflow,
    case: SdxlVisualCase,
) -> JsonObject:
    """Build complete terminal history for one declared case workflow."""

    outputs: JsonObject = {}
    for output in workflow.outputs:
        mode = _mode(output.mode)
        snapshots = [
            {
                "strategy": "attention_coupling",
                "backend": "comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel",
                "active_region_indices": [0, 1],
                "estimated_work": {
                    "cross_attention_branch_multiplier": 3.0,
                    "denoiser_call_multiplier": 1.0,
                },
                "spatial_mode": spatial_mode,
            }
            for spatial_mode in sorted(mode.expected_spatial_modes)
        ]
        outputs[output.outputs.save_node_id] = {
            "images": [
                {
                    "filename": f"{output.artifact_id}.png",
                    "subfolder": "u11",
                    "type": "output",
                }
            ]
        }
        outputs[output.outputs.metrics_node_id] = {
            "benchmark_metrics": [
                {
                    "model_call_count": (
                        SDXL_VISUAL_RUNTIME_EXPECTATIONS.expected_model_calls(
                            case, mode
                        )
                    ),
                    "runtime_ms": 10.0,
                    "peak_vram_bytes": 1,
                }
            ]
        }
        outputs[output.outputs.diagnostics_node_id] = {
            "regional_diagnostics": [
                {"record_count": len(snapshots), "snapshots": snapshots}
            ]
        }
    return {
        "status": {"completed": True, "status_str": "success", "messages": []},
        "outputs": outputs,
    }


def _mode(mode: VisualMode) -> SdxlIntegrationMode:
    """Return the fixed public sampler mode contract."""

    return next(item for item in MODES if item.mode_id == mode.value)


def _png(width: int, height: int, value: int) -> bytes:
    """Return one nonconstant exact-dimension RGB PNG."""

    image = Image.new("RGB", (width, height), (value, value, value))
    image.putpixel((0, 0), (255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
