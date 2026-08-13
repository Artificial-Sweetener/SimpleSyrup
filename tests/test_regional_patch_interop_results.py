# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify ordered P9.7 image, rejection, log, and result publication."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from tools.comfy_api import ImageReference, JsonObject
from tools.regional_patch_interop_integration.history import (
    RegionalPatchInteropError,
    RegionalPatchInteropSuccess,
)
from tools.regional_patch_interop_integration.matrix import STEPS, cases
from tools.regional_patch_interop_integration.results import (
    RegionalPatchInteropResultRecorder,
)
from tools.regional_patch_interop_integration.source_identity import (
    RepositoryRevision,
)
from tools.regional_patch_interop_integration.workflow import (
    BuiltRegionalPatchInteropWorkflow,
    RegionalPatchInteropWorkflowBuilder,
)


def test_result_recorder_publishes_labeled_image_and_zero_output_rejection(
    tmp_path: Path,
) -> None:
    """Finalize only after exact ordered artifacts, logs, and cleanup evidence."""

    accepted = cases()[0]
    rejected = next(
        case
        for case in cases()
        if case.case_id == "anima-full-scheduled-easycache-rejected"
    )
    accepted_workflow = _workflow(accepted.case_id)
    rejected_workflow = _workflow(rejected.case_id)
    recorder = RegionalPatchInteropResultRecorder(
        tmp_path,
        repositories=(RepositoryRevision("SimpleSyrup", "a" * 40, True),),
        model_inventory=({"artifact_id": "model", "sha256": "b" * 64},),
    )
    recorder.record_node_metadata({"EasyCache": {"input": {}}})
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "left.png").write_bytes(b"left")
    (input_root / "right.png").write_bytes(b"right")
    recorder.record_masks(
        {"full-1024": ("left.png", "right.png")},
        input_root=input_root,
    )
    accepted_path = recorder.record_case(
        accepted,
        accepted_workflow,
        _success(accepted_workflow),
        history={"status": {"status_str": "success"}},
        prompt_id="accepted-prompt",
        wall_runtime_ms=10.0,
        image_bytes=_png((1024, 1024)),
        source_image_bytes=None,
    )
    rejection_path = recorder.record_case(
        rejected,
        rejected_workflow,
        _rejection(rejected_workflow),
        history={"status": {"status_str": "error"}},
        prompt_id="rejected-prompt",
        wall_runtime_ms=2.0,
        image_bytes=None,
        source_image_bytes=None,
    )
    (tmp_path / "comfy.stdout.log").write_text(
        "EasyCache regional validation\n",
        encoding="utf-8",
    )
    (tmp_path / "comfy.stderr.log").write_text("", encoding="utf-8")

    result_path = recorder.finalize(
        (accepted, rejected),
        system_stats={"device": "test"},
        cleanup_verified=True,
        checkpoint_cleanup_verified=True,
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert accepted_path.name == f"{accepted.case_id}.png"
    assert rejection_path.name == f"{rejected.case_id}.rejection.json"
    assert result["status"] == "completed"
    assert [item["status"] for item in result["observations"]] == [
        "accepted",
        "rejected",
    ]
    assert result["observations"][1]["model_call_count"] == 0
    assert result["observations"][1]["image_file"] is None
    assert result["log_evidence"]["diagnostics.log"]["line_count"] == 1
    assert result["image_equivalence"] == []
    assert not (tmp_path / "p9.7-result.inprogress.json").exists()


def _workflow(case_id: str) -> BuiltRegionalPatchInteropWorkflow:
    """Build one exact Anima result fixture graph."""

    case = next(item for item in cases() if item.case_id == case_id)
    return RegionalPatchInteropWorkflowBuilder().build(
        case,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )


def _success(
    workflow: BuiltRegionalPatchInteropWorkflow,
) -> RegionalPatchInteropSuccess:
    """Return one valid full static-ADAPTER_A success value."""

    return RegionalPatchInteropSuccess(
        _snapshot(workflow, cache=False),
        {
            "run_id": workflow.metrics_run_id,
            "model_call_count": STEPS,
            "runtime_ms": 10.0,
            "peak_vram_bytes": 1,
        },
        {
            "run_id": workflow.diagnostics_run_id,
            "record_count": STEPS,
            "snapshots": [_diagnostic_snapshot() for _ in range(STEPS)],
        },
        ImageReference("result.png", "", "output"),
        None,
    )


def _rejection(
    workflow: BuiltRegionalPatchInteropWorkflow,
) -> RegionalPatchInteropError:
    """Return one valid scheduled-EasyCache rejection value."""

    return RegionalPatchInteropError(
        _snapshot(workflow, cache=True),
        workflow.sampler_node_id,
        str(workflow.prompt[workflow.sampler_node_id]["class_type"]),
        "ValueError",
        "easycache cannot compose with scheduled regional execution",
        (workflow.modifier_snapshot_node_id,),
    )


def _snapshot(
    workflow: BuiltRegionalPatchInteropWorkflow,
    *,
    cache: bool,
) -> JsonObject:
    """Return baseline or complete EasyCache upstream modifier state."""

    return {
        "run_id": f"{workflow.metrics_run_id}:modifier",
        "cache_holder_type": "EasyCacheHolder" if cache else None,
        "model_function_wrapper": None,
        "optimized_attention_override": None,
        "ppm_negpip": False,
        "wrappers": (
            [
                {
                    "wrapper_type": wrapper_type,
                    "key": "easycache",
                    "callbacks": ["callback"],
                }
                for wrapper_type in (
                    "outer_sample",
                    "calc_cond_batch",
                    "diffusion_model",
                )
            ]
            if cache
            else []
        ),
        "object_patch_keys": [],
        "transformer_patch_counts": {},
    }


def _diagnostic_snapshot() -> JsonObject:
    """Return one exact full-context regional ADAPTER_A record."""

    return {
        "strategy": "attention_coupling",
        "backend": "comfy.ldm.anima.model.Anima",
        "spatial_mode": "full",
        "adapter_uses": [
            {
                "active": True,
                "composition_index": 0,
                "region_index": 0,
                "branch": "positive",
                "target_count": 448,
                "effective_strength": 0.75,
                "adapter_token": "adapter_a-token",
            },
            {
                "active": True,
                "composition_index": 1,
                "region_index": 0,
                "branch": "negative",
                "target_count": 448,
                "effective_strength": 0.75,
                "adapter_token": "adapter_a-token",
            },
        ],
        "estimated_work": {
            "active_adapter_uses": 2,
            "active_target_count": 448,
            "target_use_count": 896,
            "denoiser_call_multiplier": 1.0,
        },
    }


def _png(size: tuple[int, int]) -> bytes:
    """Return one non-flat exact-size PNG fixture."""

    image = Image.new("RGB", size, "black")
    ImageDraw.Draw(image).rectangle((0, 0, size[0] // 2, size[1] - 1), fill="white")
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()
