"""Persist and finalize the complete pinned Prompt Control characterization."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from tools.comfy_api import JsonObject

from .cases import PromptControlCase, cases
from .evidence_validation import validate_evidence
from .history_outputs import PromptControlOutputs
from .source_identity import PromptControlSourceIdentity

BENCHMARK_ID = "prompt-control-characterization-v1"


class PromptControlResultRecorder:
    """Journal each case and finalize only a complete successful matrix."""

    def __init__(
        self,
        output_root: Path,
        source: PromptControlSourceIdentity,
        environment: JsonObject,
        matrix: tuple[PromptControlCase, ...] | None = None,
    ) -> None:
        """Initialize or resume evidence for the exact pinned source."""

        self._root = output_root.resolve()
        self._journal = self._root / "result.inprogress.json"
        self._result = self._root / "result.json"
        self._source = source
        self._environment = environment
        self._cases = matrix if matrix is not None else cases()
        self._started_at = _utc_now()
        self._observations: dict[str, JsonObject] = {}
        self._root.mkdir(parents=True, exist_ok=True)
        if self._journal.exists():
            self._resume()

    @property
    def completed_case_ids(self) -> frozenset[str]:
        """Return successful case identities eligible for resume skipping."""

        return frozenset(
            key
            for key, observation in self._observations.items()
            if observation.get("status") == "completed"
            and isinstance(observation.get("expansion"), dict)
        )

    def record_success(
        self,
        case: PromptControlCase,
        outputs: PromptControlOutputs,
        public_workflow: dict[str, JsonObject],
    ) -> None:
        """Validate and durably store one host-facing observation."""

        validate_evidence(case, outputs)
        self._observations[case.case_id] = {
            "case_id": case.case_id,
            "status": "completed",
            "public_workflow": public_workflow,
            "expansion": outputs.expansion,
            "snapshot": outputs.snapshot,
            "runtime": outputs.runtime,
            "error": None,
        }
        self._persist(self._journal, completed_at=None)

    def record_failure(self, case: PromptControlCase, error: BaseException) -> None:
        """Store a retryable failed matrix position."""

        self._observations[case.case_id] = {
            "case_id": case.case_id,
            "status": "failed",
            "public_workflow": None,
            "expansion": None,
            "snapshot": None,
            "runtime": None,
            "error": f"{type(error).__name__}: {error}",
        }
        self._persist(self._journal, completed_at=None)

    def finalize(self) -> Path:
        """Write the terminal artifact only after every case succeeds."""

        expected = {case.case_id for case in self._cases}
        if set(self._observations) != expected:
            raise ValueError("Prompt Control result is incomplete.")
        if any(
            item.get("status") != "completed" for item in self._observations.values()
        ):
            raise ValueError("Prompt Control result contains failed cases.")
        self._persist(self._result, completed_at=_utc_now())
        return self._result

    def _resume(self) -> None:
        """Restore only a journal for the same source and benchmark."""

        payload = _object(json.loads(self._journal.read_text(encoding="utf-8")))
        source = _object(payload.get("prompt_control_source"))
        if (
            payload.get("benchmark_id") != BENCHMARK_ID
            or source.get("tracked_tree_sha256") != self._source.tracked_tree_sha256
        ):
            raise ValueError("Prompt Control journal does not match the pinned source.")
        started = payload.get("started_at_utc")
        if isinstance(started, str) and started:
            self._started_at = started
        observations = payload.get("observations")
        if not isinstance(observations, list):
            raise TypeError("Prompt Control journal observations must be an array.")
        for raw in observations:
            observation = _object(raw)
            case_id = observation.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise TypeError("Prompt Control journal case_id must be text.")
            self._observations[case_id] = observation

    def _persist(self, path: Path, *, completed_at: str | None) -> None:
        """Atomically write observations in authoritative matrix order."""

        payload = {
            "schema_version": 1,
            "benchmark_id": BENCHMARK_ID,
            "started_at_utc": self._started_at,
            "completed_at_utc": completed_at,
            "prompt_control_source": {
                "version": self._source.version,
                "tracked_path_count": self._source.tracked_path_count,
                "tracked_tree_sha256": self._source.tracked_tree_sha256,
            },
            "environment": self._environment,
            "observations": [
                self._observations[case.case_id]
                for case in self._cases
                if case.case_id in self._observations
            ],
        }
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)


def _object(value: object) -> JsonObject:
    """Narrow one persisted JSON object."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError("Prompt Control result value must be an object.")
    return value


def _utc_now() -> str:
    """Return one durable UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
