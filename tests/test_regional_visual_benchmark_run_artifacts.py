# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify durable segmented-run identity and exact P10.3 resume validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from tools.attention_coupling_benchmark.manifest_types import JsonObject
from tools.regional_visual_benchmark.run_artifacts import VisualBenchmarkRunArtifacts


def _read(path: Path) -> JsonObject:
    """Read and narrow one test JSON object."""

    value: object = json.loads(path.read_text("utf-8"))
    assert isinstance(value, dict)
    return cast(JsonObject, value)


def _journal(root: Path, benchmark_id: str) -> None:
    """Write the minimum exact corpus identity accepted for resume."""

    (root / "p10.3-corpus.inprogress.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_id": benchmark_id,
                "status": "running",
                "sources": [],
                "observations": [],
            }
        ),
        encoding="utf-8",
    )


def test_create_records_one_distinct_corpus_identity(tmp_path: Path) -> None:
    """Keep corpus identity separate from independently managed process segments."""

    run = VisualBenchmarkRunArtifacts.create(tmp_path, "benchmark")

    state = _read(run.root / "p10.3-run.json")
    assert state["benchmark_id"] == "benchmark"
    assert state["run_id"] == run.root.name
    assert state["segments"] == []


def test_resume_preserves_root_and_records_clean_segment(tmp_path: Path) -> None:
    """Resume one exact journal and retain independent process cleanup evidence."""

    root = tmp_path / "existing-run"
    root.mkdir()
    _journal(root, "benchmark")
    run = VisualBenchmarkRunArtifacts.resume(root, "benchmark")
    segment = run.create_segment()
    segment.record_cleanup(process_running=False, port_available=True)
    run.record_segment_completed(
        segment,
        completed_positions=10,
        system_stats={"device": "gpu"},
    )

    state = _read(root / "p10.3-run.json")
    segments = state["segments"]
    assert isinstance(segments, list)
    assert segments[0]["status"] == "completed"
    assert segments[0]["completed_positions"] == 10
    assert run.system_stats["managed_segments"] == [
        {"segment_id": segment.run_id, "system_stats": {"device": "gpu"}}
    ]


def test_resume_rejects_another_benchmark(tmp_path: Path) -> None:
    """Never append new outputs to a journal with another frozen identity."""

    _journal(tmp_path, "other")

    with pytest.raises(ValueError, match="another benchmark"):
        VisualBenchmarkRunArtifacts.resume(tmp_path, "benchmark")
