"""Verify atomic, resumable, complete P0.7 result persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from anima_lora_characterization_fixtures import completed_outputs, inventory

from tools.anima_lora_characterization.matrix import runs
from tools.anima_lora_characterization.results import LoraResultRecorder
from tools.comfy_api import JsonObject


def test_recorder_persists_resumes_and_finalizes_exact_matrix(
    tmp_path: Path,
) -> None:
    """Journal success immediately and finalize only the authoritative set."""

    run = runs()[0]
    environment: JsonObject = {"comfyui_version": "test"}
    recorder = LoraResultRecorder(tmp_path, inventory(), environment, (run,))
    recorder.record_success(run, completed_outputs(run), b"png-bytes")

    resumed = LoraResultRecorder(tmp_path, inventory(), environment, (run,))
    assert resumed.completed_artifact_ids == {run.artifact_id}
    result_path = resumed.finalize()
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert payload["completed_at_utc"] is not None
    assert payload["observations"][0]["image_sha256"]
    assert (tmp_path / "images" / f"{run.artifact_id}.png").read_bytes() == b"png-bytes"


def test_recorder_keeps_failures_retryable(tmp_path: Path) -> None:
    """Do not classify a failed matrix position as completed on resume."""

    run = runs()[0]
    recorder = LoraResultRecorder(tmp_path, inventory(), {})
    recorder.record_failure(run, RuntimeError("failure"))

    assert recorder.completed_artifact_ids == frozenset()
    with pytest.raises(ValueError, match="incomplete"):
        recorder.finalize()
