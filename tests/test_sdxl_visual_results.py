# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify complete incremental U11 visual evidence persistence."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import cast

from PIL import Image

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.matrix import MODES, SdxlIntegrationMode
from tools.sdxl_attention_coupling_integration.visual_cases import (
    VisualMode,
    visual_cases,
)
from tools.sdxl_attention_coupling_integration.visual_history import (
    decode_sdxl_visual_history,
)
from tools.sdxl_attention_coupling_integration.visual_results import (
    SdxlVisualResultRecorder,
)
from tools.sdxl_attention_coupling_integration.visual_workflow import (
    BuiltSdxlVisualWorkflow,
    build_sdxl_visual_workflow,
)


def test_recorder_requires_all_fourteen_outputs_and_exact_cleanup(
    tmp_path: Path,
) -> None:
    """Persist every case incrementally and complete only after cleanup proof."""

    recorder = SdxlVisualResultRecorder(tmp_path)
    for index, case in enumerate(visual_cases(), 1):
        workflow = build_sdxl_visual_workflow(
            run_id="run",
            checkpoint_name=r"owned\checkpoint.safetensors",
            mask_names=("left.png", "right.png"),
            case=case,
        )
        history = _history(workflow)
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
    assert len(observations) == 14
    assert [item["artifact_id"] for item in observations if isinstance(item, dict)] == [
        f"{case.case_id}--{mode.value}"
        for case in visual_cases()
        for mode in case.modes
    ]
    first = observations[0]
    assert isinstance(first, dict)
    prompts = first["prompts"]
    assert isinstance(prompts, dict)
    assert prompts["left_positive_l"] == visual_cases()[0].left_l
    assert prompts["right_positive_g"] == visual_cases()[0].right_g


def test_failure_is_retained_without_erasing_prior_case(tmp_path: Path) -> None:
    """Keep earlier accepted evidence when a later managed case fails."""

    recorder = SdxlVisualResultRecorder(tmp_path)
    first, failed = visual_cases()[:2]
    workflow = build_sdxl_visual_workflow(
        run_id="run",
        checkpoint_name=r"owned\checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=first,
    )
    history = _history(workflow)
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


def _history(workflow: BuiltSdxlVisualWorkflow) -> JsonObject:
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
                    "model_call_count": mode.expected_model_calls,
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
