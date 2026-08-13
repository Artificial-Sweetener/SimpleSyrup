# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the closed P9.3 managed matrix and result contract."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from tools.anima_attention_coupling_workflow import (
    STEPS,
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.comfy_api import ImageReference, JsonObject
from tools.global_regional_lora_integration.matrix import cases
from tools.global_regional_lora_integration.results import (
    GlobalRegionalLoraResultRecorder,
)


def test_p9_3_matrix_covers_global_regional_distinct_and_duplicate_cases() -> None:
    """Keep clean and contaminated transition combinations explicit and ordered."""

    definitions = cases()

    assert [case.case_id for case in definitions] == [
        "global-global_adapter-regional-adapter_a-before",
        "regional-adapter_a-only",
        "global-global_adapter-regional-adapter_a-after",
        "global-adapter_a-only",
        "duplicate-global-regional-adapter_a",
    ]
    assert [len(case.integration.global_loras) for case in definitions] == [
        1,
        0,
        1,
        1,
        1,
    ]
    assert [len(case.integration.regional_loras) for case in definitions] == [
        1,
        1,
        1,
        0,
        1,
    ]
    assert [case.expect_overlap_error for case in definitions] == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_p9_3_result_requires_success_outputs_and_exact_overlap_rejection(
    tmp_path: Path,
) -> None:
    """Persist four images and one pre-sampling rejection before completion."""

    recorder = GlobalRegionalLoraResultRecorder(tmp_path)
    workflow = BuiltAnimaAttentionCouplingWorkflow(
        prompt={"save": {"class_type": "SaveImage"}},
        save_node_id="save",
        metrics_node_id="metrics",
        metrics_run_id="metrics-run",
        diagnostics_node_id="diagnostics",
        diagnostics_run_id="diagnostics-run",
    )
    definitions = cases()
    success_history: JsonObject = {
        "status": {"status_str": "success", "completed": True},
        "outputs": {"metrics": {"benchmark_metrics": [{"model_call_count": STEPS}]}},
    }
    for case in definitions[:-1]:
        color = (
            (240, 10, 10)
            if "global-global_adapter-regional" in case.case_id
            else (10, 10, 240)
            if case.case_id == "regional-adapter_a-only"
            else (10, 240, 10)
        )
        recorder.record_success(
            case,
            workflow,
            prompt_id=f"prompt-{case.case_id}",
            history=success_history,
            reference=ImageReference("image.png", "", "output"),
            image_bytes=_png(color),
        )
    rejection_history: JsonObject = {
        "status": {
            "status_str": "error",
            "completed": False,
            "messages": [
                "Regional Anima LoRA content is already applied globally to the "
                "input MODEL: 'adapter-a.safetensors'."
            ],
        },
        "outputs": {},
    }
    recorder.record_overlap_rejection(
        definitions[-1],
        workflow,
        prompt_id="prompt-duplicate",
        history=rejection_history,
    )

    result_path = recorder.finalize(
        definitions,
        system_stats={"devices": []},
        cleanup_verified=True,
        masks_removed=True,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert [item["status"] for item in result["observations"]] == [
        "success",
        "success",
        "success",
        "success",
        "rejected_before_sampling",
    ]
    assert result["transition"] == {
        "distinct_before_after": {
            "changed_pixels": 0,
            "height": 1,
            "maximum_channel_delta": 0,
            "width": 2,
        },
        "distinct_before_regional_only": {
            "changed_pixels": 2,
            "height": 1,
            "maximum_channel_delta": 230,
            "width": 2,
        },
    }


def _png(color: tuple[int, int, int]) -> bytes:
    """Encode one deterministic two-pixel RGB fixture."""

    buffer = BytesIO()
    Image.new("RGB", (2, 1), color).save(buffer, format="PNG")
    return buffer.getvalue()
