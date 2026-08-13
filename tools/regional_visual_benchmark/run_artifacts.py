# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own durable identity and segment evidence for one P10.3 corpus run."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from tools.attention_coupling_benchmark.manifest_types import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts


class VisualBenchmarkRunArtifacts:
    """Persist one resumable corpus identity and its managed process segments."""

    def __init__(self, root: Path, benchmark_id: str, *, resumed: bool) -> None:
        """Validate one existing root or initialize its focused run record."""

        self.root = root.resolve()
        self.run_id = self.root.name
        self._benchmark_id = benchmark_id
        self._state_path = self.root / "p10.3-run.json"
        self._segments_root = self.root / "managed-segments"
        if resumed:
            self._validate_resume_root()
        self._state = self._load_or_initialize(resumed=resumed)

    @classmethod
    def create(
        cls, output_root: Path, benchmark_id: str
    ) -> VisualBenchmarkRunArtifacts:
        """Create one collision-resistant external corpus directory."""

        run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        root = output_root.resolve() / run_id
        root.mkdir(parents=True, exist_ok=False)
        return cls(root, benchmark_id, resumed=False)

    @classmethod
    def resume(cls, root: Path, benchmark_id: str) -> VisualBenchmarkRunArtifacts:
        """Open one exact incomplete corpus without changing its identity."""

        return cls(root, benchmark_id, resumed=True)

    def create_segment(self) -> IntegrationArtifacts:
        """Create one independently logged managed-Comfy process artifact."""

        segment = IntegrationArtifacts(self._segments_root)
        segments = self._segments()
        segments.append(
            {
                "segment_id": segment.run_id,
                "artifact_directory": segment.root.relative_to(self.root).as_posix(),
                "status": "starting",
            }
        )
        self._state["status"] = "running"
        self._persist()
        return segment

    def record_segment_completed(
        self,
        segment: IntegrationArtifacts,
        *,
        completed_positions: int,
        system_stats: JsonObject,
    ) -> None:
        """Record one clean bounded process lifetime and its completed work."""

        record = self._segment_record(segment.run_id)
        record.update(
            {
                "status": "completed",
                "completed_positions": completed_positions,
                "system_stats": system_stats,
                "completed_at_utc": _utc_now(),
            }
        )
        self._persist()

    def record_segment_failed(
        self, segment: IntegrationArtifacts, error: BaseException
    ) -> None:
        """Retain one failed segment without altering recorded corpus positions."""

        record = self._segment_record(segment.run_id)
        record.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "completed_at_utc": _utc_now(),
            }
        )
        self._persist()

    def record_completed(self, result_path: Path) -> None:
        """Finalize the run only after the complete corpus and blind packet exist."""

        self._state.update(
            {
                "status": "completed",
                "result_file": result_path.resolve().relative_to(self.root).as_posix(),
                "completed_at_utc": _utc_now(),
            }
        )
        self._persist()

    def record_failure(self, error: BaseException) -> None:
        """Persist an actionable run failure while keeping resume state intact."""

        self._state.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "completed_at_utc": _utc_now(),
            }
        )
        self._persist()

    @property
    def system_stats(self) -> JsonObject:
        """Return per-segment environment evidence for corpus publication."""

        return {
            "managed_segments": [
                {
                    "segment_id": record.get("segment_id"),
                    "system_stats": record.get("system_stats"),
                }
                for record in self._segments()
                if record.get("status") == "completed"
            ]
        }

    def _validate_resume_root(self) -> None:
        """Require one existing in-progress or completed corpus of this benchmark."""

        if not self.root.is_dir():
            raise ValueError(f"P10.3 resume root does not exist: {self.root}")
        corpus_path = self.root / "p10.3-corpus.inprogress.json"
        completed_path = self.root / "p10.3-corpus.json"
        evidence_path = corpus_path if corpus_path.is_file() else completed_path
        if not evidence_path.is_file():
            raise ValueError("P10.3 resume root has no corpus journal or result.")
        payload = _object(json.loads(evidence_path.read_text("utf-8")), "corpus")
        if payload.get("benchmark_id") != self._benchmark_id:
            raise ValueError("P10.3 resume root belongs to another benchmark.")

    def _load_or_initialize(self, *, resumed: bool) -> JsonObject:
        """Load exact prior state or create the first focused run record."""

        if self._state_path.is_file():
            state = _object(
                json.loads(self._state_path.read_text("utf-8")), "run state"
            )
            if state.get("benchmark_id") != self._benchmark_id:
                raise ValueError("P10.3 run state belongs to another benchmark.")
            if state.get("run_id") != self.run_id:
                raise ValueError("P10.3 run state identity does not match its root.")
            state["status"] = "resuming" if resumed else "starting"
            state["resumed_at_utc"] = _utc_now()
        else:
            state = {
                "schema_version": 1,
                "benchmark_id": self._benchmark_id,
                "run_id": self.run_id,
                "status": "resuming" if resumed else "starting",
                "started_at_utc": _utc_now(),
                "segments": [],
            }
        self._state = state
        self._persist()
        return state

    def _segments(self) -> list[JsonObject]:
        """Return the mutable typed segment ledger."""

        value = self._state.get("segments")
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            raise TypeError("P10.3 run segments must be an object array.")
        return cast(list[JsonObject], value)

    def _segment_record(self, segment_id: str) -> JsonObject:
        """Return one exact created segment record."""

        for record in self._segments():
            if record.get("segment_id") == segment_id:
                return record
        raise KeyError(f"P10.3 managed segment is unknown: {segment_id}")

    def _persist(self) -> None:
        """Atomically publish the authoritative segmented run record."""

        temporary = self._state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._state_path)


def _object(value: object, label: str) -> JsonObject:
    """Narrow one JSON object with string keys."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"P10.3 {label} must be an object.")
    return cast(JsonObject, value)


def _utc_now() -> str:
    """Return one durable UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
