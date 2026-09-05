# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify complete P9.1 result journaling, artifacts, and cleanup gate."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest
from PIL import Image
from prompt_control_attention_coupling_values import evidence_outputs

from tools.comfy_api import ImageReference, JsonObject
from tools.prompt_control_attention_coupling_integration.baseline import load_baseline
from tools.prompt_control_attention_coupling_integration.matrix import cases
from tools.prompt_control_attention_coupling_integration.results import (
    PromptControlAttentionResultRecorder,
)
from tools.prompt_control_attention_coupling_integration.workflow import (
    PromptControlAttentionWorkflowBuilder,
)
from tools.prompt_control_characterization.source_identity import (
    PINNED_TRACKED_PATH_COUNT,
    PINNED_TRACKED_TREE_SHA256,
    PINNED_VERSION,
    PromptControlSourceIdentity,
)

pytestmark = pytest.mark.external_artifact


def test_result_recorder_requires_complete_ordered_matrix_and_cleanup(
    tmp_path: Path,
) -> None:
    """Journal every labeled output and finalize only complete managed evidence."""

    baseline = load_baseline()
    source = PromptControlSourceIdentity(
        PINNED_VERSION,
        PINNED_TRACKED_PATH_COUNT,
        PINNED_TRACKED_TREE_SHA256,
    )
    recorder = PromptControlAttentionResultRecorder(tmp_path, baseline, source)
    input_root = tmp_path / "input"
    input_root.mkdir()
    mask_names = ("left.png", "right.png")
    for name in mask_names:
        (input_root / name).write_bytes(name.encode())
    recorder.record_masks(mask_names, input_root=input_root)
    recorder.record_metadata({"name": "SimpleSyrup.KSamplerAttentionCoupling"})
    builder = PromptControlAttentionWorkflowBuilder()
    image_bytes = _png()
    definitions = cases()
    for index, case in enumerate(definitions):
        workflow = builder.build(
            case,
            run_id="run",
            mask_names=mask_names,
        )
        outputs = evidence_outputs(case, workflow, baseline.observation(case.case_id))
        recorder.record_case(
            case,
            workflow,
            outputs,
            history={"case_id": case.case_id},
            prompt_id=f"prompt-{index}",
            image_reference=ImageReference(f"{case.case_id}.png", "p91", "output"),
            image_bytes=image_bytes,
            wall_runtime_ms=100.0,
        )
    assert (tmp_path / "p9.1-result.inprogress.json").is_file()
    with pytest.raises(ValueError, match="cleanup"):
        recorder.finalize(definitions, system_stats={}, cleanup_verified=False)

    result_path = recorder.finalize(
        definitions,
        system_stats={"system": {}},
        cleanup_verified=True,
    )
    payload = cast(JsonObject, json.loads(result_path.read_text(encoding="utf-8")))
    observations = cast(list[JsonObject], payload["observations"])
    assert payload["status"] == "completed"
    assert len(observations) == 9
    assert {
        cast(JsonObject, item["metrics"])["model_call_count"] for item in observations
    } == {8}
    assert not (tmp_path / "p9.1-result.inprogress.json").exists()


def _png() -> bytes:
    """Return one exact-size non-flat PNG fixture."""

    image = Image.new("RGB", (512, 512), color=(0, 0, 0))
    image.putpixel((0, 0), (255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
