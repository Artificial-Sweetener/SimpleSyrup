# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove ordered durable P9.6 rejection-sidecar publication."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from anima_nondiffusion_rejection_values import synthetic_rejection

from tools.anima_regional_lora_admission_integration.workflow import (
    RegionalLoraAdmissionWorkflowBuilder,
)
from tools.anima_regional_nondiffusion_rejection_integration.matrix import cases
from tools.anima_regional_nondiffusion_rejection_integration.results import (
    AnimaNondiffusionRejectionResultRecorder,
)


def test_recorder_publishes_four_labeled_rejections_and_no_image(
    tmp_path: Path,
) -> None:
    """Persist the complete ordered matrix only after cleanup succeeds."""

    definitions = cases()
    recorder = AnimaNondiffusionRejectionResultRecorder(tmp_path)
    builder = RegionalLoraAdmissionWorkflowBuilder(artifact_phase="p9.6")
    for case in definitions:
        workflow = builder.build(
            case,
            run_id="run",
            mask_names=("left.png", "right.png"),
        )
        recorder.record_case(
            case,
            workflow,
            synthetic_rejection(case, workflow),
            history={"outputs": {}, "status": {}},
            prompt_id=f"prompt-{case.case_id}",
            wall_runtime_ms=1.0,
        )

    with pytest.raises(ValueError, match="cleanup failed"):
        recorder.finalize(definitions, system_stats={}, cleanup_verified=False)
    result = recorder.finalize(definitions, system_stats={}, cleanup_verified=True)
    payload = json.loads(result.read_text(encoding="utf-8"))

    assert payload["phase"] == "p9.6"
    assert payload["status"] == "completed"
    assert payload["turbo_artifact"]["sha256"] == (
        "1b55e40bdb1d0e5a78cb498f245fccfdaae97823265db957d2aabdcf4cd3caf1"
    )
    assert [item["issue_count"] for item in payload["observations"]] == [60, 1, 2, 2]
    assert len(list(tmp_path.glob("*.rejection.json"))) == 4
    assert list(tmp_path.glob("*.png")) == []


def test_recorder_rejects_duplicate_case_publication(tmp_path: Path) -> None:
    """Keep every labeled rejection sidecar unique and ordered."""

    case = cases()[1]
    workflow = RegionalLoraAdmissionWorkflowBuilder(artifact_phase="p9.6").build(
        case,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )
    observed = synthetic_rejection(case, workflow)
    recorder = AnimaNondiffusionRejectionResultRecorder(tmp_path)
    recorder.record_case(
        case,
        workflow,
        observed,
        history={},
        prompt_id="prompt",
        wall_runtime_ms=1.0,
    )

    with pytest.raises(ValueError, match="already recorded"):
        recorder.record_case(
            case,
            workflow,
            observed,
            history={},
            prompt_id="prompt",
            wall_runtime_ms=1.0,
        )
