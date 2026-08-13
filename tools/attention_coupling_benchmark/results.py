# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Parse and persist complete Attention Coupling benchmark evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tools.comfy_api import ImageReference

from .manifest import DEFAULT_MANIFEST_PATH
from .manifest_decode import json_array, json_object
from .manifest_types import BenchmarkManifest, BenchmarkRun, JsonObject


@dataclass(frozen=True)
class ProbeMetrics:
    """Hold exact measurements returned by the benchmark-only Comfy probe."""

    runtime_ms: float
    peak_vram_bytes: int
    model_call_count: int


@dataclass(frozen=True)
class CompletedOutputs:
    """Hold the probe metrics and sole saved image from prompt history."""

    metrics: ProbeMetrics
    image: ImageReference


def parse_completed_outputs(
    history: JsonObject,
    *,
    metrics_node_id: str,
    save_node_id: str,
) -> CompletedOutputs:
    """Extract exact probe and image outputs or surface the server failure."""

    status = json_object(history.get("status"), "history.status")
    if status.get("status_str") != "success":
        raise RuntimeError(f"ComfyUI execution failed: {status.get('messages')!r}.")
    outputs = json_object(history.get("outputs"), "history.outputs")
    metrics_output = json_object(outputs.get(metrics_node_id), "probe output")
    metric_entries = json_array(
        metrics_output.get("benchmark_metrics"), "probe benchmark_metrics"
    )
    if len(metric_entries) != 1:
        raise ValueError("Benchmark probe must return exactly one metric record.")
    metric_values = json_object(metric_entries[0], "probe metric record")
    metrics = ProbeMetrics(
        runtime_ms=_number(metric_values.get("runtime_ms"), "runtime_ms"),
        peak_vram_bytes=_integer(
            metric_values.get("peak_vram_bytes"), "peak_vram_bytes"
        ),
        model_call_count=_integer(
            metric_values.get("model_call_count"), "model_call_count"
        ),
    )
    save_output = json_object(outputs.get(save_node_id), "save output")
    images = json_array(save_output.get("images"), "saved images")
    if len(images) != 1:
        raise ValueError("Benchmark workflow must save exactly one image.")
    image = json_object(images[0], "saved image")
    return CompletedOutputs(
        metrics=metrics,
        image=ImageReference(
            filename=_text(image.get("filename"), "saved image filename"),
            subfolder=_text(
                image.get("subfolder"), "saved image subfolder", empty=True
            ),
            output_type=_text(image.get("type"), "saved image type"),
        ),
    )


class BenchmarkResultRecorder:
    """Persist each observation immediately and finalize only a complete matrix."""

    def __init__(
        self,
        manifest: BenchmarkManifest,
        output_root: Path,
        environment: JsonObject,
    ) -> None:
        """Initialize or resume one exact manifest result set."""

        self._manifest = manifest
        self._root = output_root.resolve()
        self._images = self._root / "images"
        self._journal = self._root / "result.inprogress.json"
        self._result = self._root / "result.json"
        self._root.mkdir(parents=True, exist_ok=True)
        self._images.mkdir(parents=True, exist_ok=True)
        self._manifest_sha256 = _sha256(DEFAULT_MANIFEST_PATH.read_bytes())
        self._environment = environment
        self._started_at = _utc_now()
        self._observations: dict[str, JsonObject] = {}
        if self._journal.exists():
            self._resume()

    @property
    def completed_artifact_ids(self) -> frozenset[str]:
        """Return successful artifact IDs already preserved in the journal."""

        return frozenset(
            artifact_id
            for artifact_id, observation in self._observations.items()
            if observation["status"] == "completed"
        )

    def record_success(
        self,
        run: BenchmarkRun,
        outputs: CompletedOutputs,
        image_bytes: bytes,
    ) -> None:
        """Preserve one image and its successful measured observation."""

        image_path = self._images / f"{run.artifact_id}.png"
        image_path.write_bytes(image_bytes)
        self._observations[run.artifact_id] = {
            "artifact_id": run.artifact_id,
            "case_id": run.case_id,
            "execution_id": run.execution_id,
            "seed": run.seed,
            "status": "completed",
            "output_sha256": _sha256(image_bytes),
            "runtime_ms": outputs.metrics.runtime_ms,
            "peak_vram_bytes": outputs.metrics.peak_vram_bytes,
            "model_call_count": outputs.metrics.model_call_count,
            "cross_attention_branch_count": None,
            "scores": None,
            "failure_classes": [],
            "notes": "",
            "error": None,
        }
        self._persist(self._journal, completed_at=None)

    def record_failure(self, run: BenchmarkRun, error: BaseException) -> None:
        """Preserve one failed run without omitting its matrix position."""

        self._observations[run.artifact_id] = {
            "artifact_id": run.artifact_id,
            "case_id": run.case_id,
            "execution_id": run.execution_id,
            "seed": run.seed,
            "status": "failed",
            "output_sha256": None,
            "runtime_ms": None,
            "peak_vram_bytes": None,
            "model_call_count": None,
            "cross_attention_branch_count": None,
            "scores": None,
            "failure_classes": [],
            "notes": "",
            "error": f"{type(error).__name__}: {error}",
        }
        self._persist(self._journal, completed_at=None)

    def finalize(self) -> Path:
        """Write the final result only when every manifest run is successful."""

        expected = {run.artifact_id for run in self._manifest.runs()}
        actual = set(self._observations)
        if actual != expected:
            missing_count = len(expected - actual)
            unexpected_count = len(actual - expected)
            raise ValueError(
                "Benchmark result is incomplete: "
                f"missing={missing_count}, unexpected={unexpected_count}."
            )
        failed = [
            artifact_id
            for artifact_id, observation in self._observations.items()
            if observation["status"] != "completed"
        ]
        if failed:
            raise ValueError(f"Benchmark has {len(failed)} failed runs; retry them.")
        self._persist(self._result, completed_at=_utc_now())
        return self._result

    def _resume(self) -> None:
        """Restore observations only from the same manifest identity."""

        payload: object = json.loads(self._journal.read_text(encoding="utf-8"))
        root = json_object(payload, "benchmark journal")
        if root.get("benchmark_id") != self._manifest.benchmark_id:
            raise ValueError("Benchmark journal belongs to a different benchmark ID.")
        if root.get("manifest_sha256") != self._manifest_sha256:
            raise ValueError(
                "Benchmark journal manifest hash does not match current inputs."
            )
        started_at = root.get("started_at_utc")
        if isinstance(started_at, str) and started_at:
            self._started_at = started_at
        for item in json_array(root.get("observations"), "journal observations"):
            observation = json_object(item, "journal observation")
            artifact_id = _text(observation.get("artifact_id"), "artifact_id")
            self._observations[artifact_id] = observation

    def _persist(self, path: Path, *, completed_at: str | None) -> None:
        """Atomically persist sorted observations in manifest expansion order."""

        payload: JsonObject = {
            "schema_version": 1,
            "benchmark_id": self._manifest.benchmark_id,
            "manifest_sha256": self._manifest_sha256,
            "run_id": self._root.name,
            "started_at_utc": self._started_at,
            "completed_at_utc": completed_at,
            "environment": self._environment,
            "observations": [
                self._observations[run.artifact_id]
                for run in self._manifest.runs()
                if run.artifact_id in self._observations
            ],
        }
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def _sha256(value: bytes) -> str:
    """Return a lowercase SHA-256 identity."""

    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    """Return a JSON-schema-compatible UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _integer(value: object, label: str) -> int:
    """Narrow one nonnegative metric integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer.")
    return value


def _number(value: object, label: str) -> float:
    """Narrow one nonnegative finite metric number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a nonnegative number.")
    result = float(value)
    if result < 0 or result != result or result == float("inf"):
        raise ValueError(f"{label} must be a nonnegative finite number.")
    return result


def _text(value: object, label: str, *, empty: bool = False) -> str:
    """Narrow one string metric or image field."""

    if not isinstance(value, str) or (not empty and not value):
        raise ValueError(f"{label} must be a string.")
    return value
