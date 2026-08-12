"""Verify benchmark history parsing and complete result persistence."""

import json
from pathlib import Path

import pytest

from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.attention_coupling_benchmark.results import (
    BenchmarkResultRecorder,
    CompletedOutputs,
    ProbeMetrics,
    parse_completed_outputs,
)
from tools.comfy_api import ImageReference


def test_completed_history_exposes_probe_and_saved_image() -> None:
    """Narrow the exact metric and image fields emitted by the workflow."""

    outputs = parse_completed_outputs(
        {
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {
                "7": {
                    "benchmark_metrics": [
                        {
                            "runtime_ms": 123.5,
                            "peak_vram_bytes": 456,
                            "model_call_count": 78,
                        }
                    ]
                },
                "9": {
                    "images": [
                        {
                            "filename": "image.png",
                            "subfolder": "bench",
                            "type": "output",
                        }
                    ]
                },
            },
        },
        metrics_node_id="7",
        save_node_id="9",
    )

    assert outputs.metrics == ProbeMetrics(123.5, 456, 78)
    assert outputs.image == ImageReference("image.png", "bench", "output")


def test_recorder_finalizes_only_complete_successful_matrix(tmp_path: Path) -> None:
    """Persist every manifest position and validate the final result shape."""

    manifest = load_manifest()
    environment: dict[str, object] = {
        "simple_syrup_commit": manifest.simple_syrup_commit,
        "comfyui_commit": manifest.comfyui_commit,
        "python_version": "3.12.10",
        "torch_version": "test",
        "cuda_version": "13.0",
        "gpu_name": "test gpu",
        "attention_backend": "attention_pytorch",
        "model_hashes": {model.artifact_id: model.sha256 for model in manifest.models},
    }
    recorder = BenchmarkResultRecorder(manifest, tmp_path / "result", environment)
    completed = CompletedOutputs(
        metrics=ProbeMetrics(1.0, 2, 3),
        image=ImageReference("server.png", "", "output"),
    )
    for run in manifest.runs():
        recorder.record_success(run, completed, b"png")

    result_path = recorder.finalize()
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert payload["completed_at_utc"].endswith("Z")
    assert len(payload["observations"]) == 180
    assert {item["status"] for item in payload["observations"]} == {"completed"}
    assert len(list((tmp_path / "result" / "images").glob("*.png"))) == 180


def test_recorder_retries_failed_journal_positions(tmp_path: Path) -> None:
    """Keep failures visible while allowing the same position to be corrected."""

    manifest = load_manifest()
    recorder = BenchmarkResultRecorder(manifest, tmp_path / "result", {})
    run = manifest.runs()[0]

    recorder.record_failure(run, RuntimeError("failure"))

    resumed = BenchmarkResultRecorder(manifest, tmp_path / "result", {})
    assert run.artifact_id not in resumed.completed_artifact_ids
    with pytest.raises(ValueError, match="incomplete"):
        resumed.finalize()
