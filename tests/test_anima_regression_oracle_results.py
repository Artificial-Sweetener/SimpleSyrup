# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify atomic terminal Anima oracle result publication."""

from __future__ import annotations

import json
from pathlib import Path

from tools.anima_regression_oracle.artifact_validation import ArtifactObservation
from tools.anima_regression_oracle.execution import CommandObservation
from tools.anima_regression_oracle.results import AnimaRegressionResultRecorder


def test_result_is_completed_only_for_complete_passing_command_set(
    tmp_path: Path,
) -> None:
    """Publish a complete ordered result with no leftover journal."""

    recorder = AnimaRegressionResultRecorder(tmp_path / "passing")

    result = recorder.publish(
        level="focused",
        artifacts=(ArtifactObservation("image", "image.png", "a" * 64),),
        commands=(CommandObservation("focused", ("python",), 0, 1.0, "out", "err"),),
        expected_command_count=1,
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["expected_command_count"] == 1
    assert payload["artifact_observations"][0]["identity"] == "image"
    assert payload["command_observations"][0]["identity"] == "focused"
    assert not result.with_suffix(".json.tmp").exists()


def test_result_retains_failure_or_incomplete_execution(tmp_path: Path) -> None:
    """Never publish completion from a failed or truncated gate set."""

    recorder = AnimaRegressionResultRecorder(tmp_path / "failed")

    result = recorder.publish(
        level="full",
        artifacts=(),
        commands=(CommandObservation("focused", ("python",), 1, 1.0, "out", "err"),),
        expected_command_count=5,
    )

    assert json.loads(result.read_text(encoding="utf-8"))["status"] == "failed"
