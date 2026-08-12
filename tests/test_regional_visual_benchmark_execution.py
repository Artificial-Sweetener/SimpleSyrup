"""Characterize bounded P10.3 server segmentation and exact journal resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.regional_visual_benchmark.execution import VisualCorpusExecutor
from tools.regional_visual_benchmark.matrix import source_positions, visual_positions
from tools.regional_visual_benchmark.run_artifacts import VisualBenchmarkRunArtifacts


def _executor(root: Path, *, segment_size: int = 10) -> VisualCorpusExecutor:
    """Build one executor without starting external state."""

    manifest = load_manifest()
    return VisualCorpusExecutor(
        comfy_root=Path("<COMFY_ROOT>"),
        run_artifacts=VisualBenchmarkRunArtifacts.create(root, manifest.benchmark_id),
        manifest=manifest,
        readiness_timeout=1.0,
        prompt_timeout=1.0,
        segment_size=segment_size,
    )


def test_bounded_execution_restarts_after_at_most_ten_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Split ordered corpus work into clean process lifetimes of ten or fewer."""

    executor = _executor(tmp_path)
    segments: list[tuple[int, int]] = []
    monkeypatch.setattr(
        executor,
        "_execute_segment",
        lambda sources, visuals: segments.append((len(sources), len(visuals))),
    )

    executor.execute(max_sources=25, max_visuals=0)

    assert segments == [(10, 0), (10, 0), (5, 0)]


def test_resume_skips_exact_recorded_source_and_visual(tmp_path: Path) -> None:
    """Never reroll a durable journal entry when resuming the same corpus."""

    manifest = load_manifest()
    sources = source_positions(manifest)
    visuals = visual_positions(manifest)
    root = tmp_path / "existing-run"
    root.mkdir()
    (root / "p10.3-corpus.inprogress.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_id": manifest.benchmark_id,
                "status": "running",
                "sources": [{"source_id": sources[0].source_id}],
                "observations": [{"artifact_id": visuals[0].artifact_id}],
            }
        ),
        encoding="utf-8",
    )
    executor = VisualCorpusExecutor(
        comfy_root=Path("<COMFY_ROOT>"),
        run_artifacts=VisualBenchmarkRunArtifacts.resume(root, manifest.benchmark_id),
        manifest=manifest,
        readiness_timeout=1.0,
        prompt_timeout=1.0,
    )

    pending = executor.pending_work(max_sources=1, max_visuals=1)

    assert pending.sources == ()
    assert pending.visuals == ()


def test_segment_size_must_be_positive(tmp_path: Path) -> None:
    """Reject a restart policy that could make no forward progress."""

    with pytest.raises(ValueError, match="at least one"):
        _executor(tmp_path, segment_size=0)
