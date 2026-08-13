# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove ordered durable P9.5 result publication and cleanup gating."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from tools.anima_regional_lora_admission_integration.history import (
    RegionalLoraAdmissionError,
    RegionalLoraAdmissionHistory,
    RegionalLoraAdmissionSuccess,
)
from tools.anima_regional_lora_admission_integration.matrix import (
    RegionalLoraAdmissionCase,
    cases,
)
from tools.anima_regional_lora_admission_integration.results import (
    RegionalLoraAdmissionResultRecorder,
)
from tools.anima_regional_lora_admission_integration.workflow import (
    BuiltRegionalLoraAdmissionWorkflow,
    RegionalLoraAdmissionWorkflowBuilder,
)
from tools.comfy_api import ImageReference


def test_recorder_publishes_one_image_and_six_exact_rejections(tmp_path: Path) -> None:
    """Persist the complete ordered matrix only after cleanup succeeds."""

    definitions = cases()
    recorder = RegionalLoraAdmissionResultRecorder(tmp_path)
    for case in definitions:
        workflow = _workflow(case)
        observed = _observed(case, workflow)
        recorder.record_case(
            case,
            workflow,
            observed,
            history={"outputs": {}, "status": {}},
            prompt_id=f"prompt-{case.case_id}",
            wall_runtime_ms=1.0,
            image_reference=(
                ImageReference("image.png", "", "output")
                if case.expect_success
                else None
            ),
            image_bytes=_image_bytes() if case.expect_success else None,
        )

    with pytest.raises(ValueError, match="cleanup failed"):
        recorder.finalize(definitions, system_stats={}, cleanup_verified=False)
    result = recorder.finalize(definitions, system_stats={}, cleanup_verified=True)
    payload = json.loads(result.read_text(encoding="utf-8"))

    assert payload["status"] == "completed"
    assert [item["status"] for item in payload["observations"]] == [
        "success",
        "rejected",
        "rejected",
        "rejected",
        "rejected",
        "rejected",
        "rejected",
    ]
    assert (tmp_path / "supported-adapter_a.png").is_file()
    assert list(tmp_path.glob("*.png")) == [tmp_path / "supported-adapter_a.png"]


def test_recorder_rejects_images_on_failure_cases(tmp_path: Path) -> None:
    """Prevent a partially sampled failure from becoming accepted evidence."""

    case = cases()[1]
    workflow = _workflow(case)
    recorder = RegionalLoraAdmissionResultRecorder(tmp_path)

    with pytest.raises(ValueError, match="must not publish an image"):
        recorder.record_case(
            case,
            workflow,
            _observed(case, workflow),
            history={},
            prompt_id="prompt",
            wall_runtime_ms=1.0,
            image_reference=ImageReference("image.png", "", "output"),
            image_bytes=_image_bytes(),
        )


def _workflow(
    case: RegionalLoraAdmissionCase,
) -> BuiltRegionalLoraAdmissionWorkflow:
    """Build one exact synthetic managed graph."""

    return RegionalLoraAdmissionWorkflowBuilder(artifact_phase="p9.5").build(
        case,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )


def _observed(
    case: RegionalLoraAdmissionCase,
    workflow: BuiltRegionalLoraAdmissionWorkflow,
) -> RegionalLoraAdmissionHistory:
    """Return already narrowed exact evidence for one case."""

    if case.expect_success:
        snapshot = {
            "strategy": "attention_coupling",
            "backend": "comfy.ldm.anima.model.Anima",
            "spatial_mode": "full",
            "adapter_uses": [
                {
                    "active": True,
                    "adapter_token": "token",
                    "branch": "positive",
                    "composition_index": 0,
                    "effective_strength": 0.75,
                    "region_index": 0,
                    "target_count": 448,
                }
            ],
            "estimated_work": {
                "active_target_count": 448,
                "denoiser_call_multiplier": 1.0,
            },
        }
        return RegionalLoraAdmissionSuccess(
            {
                "run_id": workflow.workflow.metrics_run_id,
                "model_call_count": 8,
                "runtime_ms": 1.0,
                "peak_vram_bytes": 1,
            },
            {
                "run_id": workflow.workflow.diagnostics_run_id,
                "record_count": 8,
                "snapshots": [snapshot.copy() for _ in range(8)],
            },
        )
    suffix = case.expected_exception_suffix
    assert suffix is not None
    return RegionalLoraAdmissionError(
        workflow.sampler_node_id,
        "SimpleSyrup.KSamplerAttentionCoupling",
        f"fixture.{suffix}",
        "\n".join(case.expected_error_fragments),
        ("1", "2"),
    )


def _image_bytes() -> bytes:
    """Return one deterministic non-flat 512-square PNG."""

    image = Image.new("RGB", (512, 512), "black")
    for x in range(256, 512):
        for y in range(512):
            image.putpixel((x, y), (255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
