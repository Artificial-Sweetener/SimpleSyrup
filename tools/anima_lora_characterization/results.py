"""Journal and finalize complete validated ADAPTER_A characterization results."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from tools.comfy_api import JsonObject

from .artifact_inventory import AdapterInventory
from .evidence_validation import validate_run_evidence
from .history_outputs import LoraCompletedOutputs
from .json_contract import array_value, object_value, text_value
from .matrix import LoraRun, runs

BENCHMARK_ID = "adapter_a-global-characterization-v1"


class LoraResultRecorder:
    """Journal every run and finalize only the complete validated matrix."""

    def __init__(
        self,
        output_root: Path,
        inventory: AdapterInventory,
        environment: JsonObject,
        run_matrix: tuple[LoraRun, ...] | None = None,
    ) -> None:
        """Initialize or resume evidence for the exact matrix and adapter."""

        self._root = output_root.resolve()
        self._images = self._root / "images"
        self._journal = self._root / "result.inprogress.json"
        self._result = self._root / "result.json"
        self._inventory = inventory
        self._environment = environment
        self._runs = run_matrix if run_matrix is not None else runs()
        self._matrix_sha256 = _matrix_sha256(self._runs)
        self._started_at = _utc_now()
        self._observations: dict[str, JsonObject] = {}
        self._images.mkdir(parents=True, exist_ok=True)
        if self._journal.exists():
            self._resume()

    @property
    def completed_artifact_ids(self) -> frozenset[str]:
        """Return only successful run identities eligible for resume skipping."""

        return frozenset(
            key
            for key, value in self._observations.items()
            if value.get("status") == "completed"
        )

    def record_success(
        self, run: LoraRun, outputs: LoraCompletedOutputs, image_bytes: bytes
    ) -> None:
        """Validate and durably record one successful run and image."""

        validate_run_evidence(run, outputs, self._inventory)
        (self._images / f"{run.artifact_id}.png").write_bytes(image_bytes)
        self._observations[run.artifact_id] = {
            **_run_fields(run),
            "status": "completed",
            "image_sha256": _sha256(image_bytes),
            "metrics": outputs.metrics,
            "error": None,
        }
        self._persist(self._journal, completed_at=None)

    def record_failure(self, run: LoraRun, error: BaseException) -> None:
        """Record a retryable failure without omitting its matrix position."""

        self._observations[run.artifact_id] = {
            **_run_fields(run),
            "status": "failed",
            "image_sha256": None,
            "metrics": None,
            "error": f"{type(error).__name__}: {error}",
        }
        self._persist(self._journal, completed_at=None)

    def finalize(self) -> Path:
        """Write the terminal result only after all 28 runs succeed."""

        expected = {run.artifact_id for run in self._runs}
        actual = set(self._observations)
        if actual != expected:
            raise ValueError(
                "LoRA result incomplete: "
                f"missing={len(expected - actual)}, "
                f"unexpected={len(actual - expected)}."
            )
        failed = [
            key
            for key, value in self._observations.items()
            if value.get("status") != "completed"
        ]
        if failed:
            raise ValueError(f"LoRA result has {len(failed)} failed runs; retry them.")
        self._persist(self._result, completed_at=_utc_now())
        return self._result

    def _resume(self) -> None:
        """Restore only a journal with the same matrix and adapter identities."""

        root = object_value(
            json.loads(self._journal.read_text(encoding="utf-8")), "LoRA journal"
        )
        if (
            root.get("benchmark_id") != BENCHMARK_ID
            or root.get("matrix_sha256") != self._matrix_sha256
        ):
            raise ValueError("LoRA journal does not match the current matrix.")
        adapter = object_value(root.get("adapter"), "journal adapter")
        if adapter.get("sha256") != self._inventory.sha256:
            raise ValueError("LoRA journal does not match the pinned adapter.")
        started = root.get("started_at_utc")
        if isinstance(started, str) and started:
            self._started_at = started
        for item in array_value(root.get("observations"), "journal observations"):
            observation = object_value(item, "journal observation")
            artifact_id = text_value(observation.get("artifact_id"), "artifact_id")
            self._observations[artifact_id] = observation

    def _persist(self, path: Path, *, completed_at: str | None) -> None:
        """Atomically write observations in authoritative matrix order."""

        payload: JsonObject = {
            "schema_version": 1,
            "benchmark_id": BENCHMARK_ID,
            "matrix_sha256": self._matrix_sha256,
            "started_at_utc": self._started_at,
            "completed_at_utc": completed_at,
            "adapter": {
                "stable_name": "adapter-a.safetensors",
                "size_bytes": self._inventory.size_bytes,
                "sha256": self._inventory.sha256,
                "pair_count": len(self._inventory.pairs),
                "target_keys": list(self._inventory.target_keys),
            },
            "environment": self._environment,
            "observations": [
                self._observations[run.artifact_id]
                for run in self._runs
                if run.artifact_id in self._observations
            ],
        }
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)


def _run_fields(run: LoraRun) -> JsonObject:
    """Serialize immutable profile fields without local model paths."""

    return {
        "artifact_id": run.artifact_id,
        "profile_id": run.profile.profile_id,
        "mode": run.profile.mode,
        "seed": run.seed,
        "capture_outputs": run.capture_outputs,
        "adapters": [
            {
                "identity": adapter.identity,
                "strength": adapter.strength,
                "schedule": [list(item) for item in adapter.schedule],
            }
            for adapter in run.profile.adapters
        ],
    }


def _matrix_sha256(run_matrix: tuple[LoraRun, ...]) -> str:
    """Hash the exact expanded matrix contract for safe resume."""

    payload = [_run_fields(run) for run in run_matrix]
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _sha256(value: bytes) -> str:
    """Return a lowercase SHA-256 identity."""

    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    """Return one UTC timestamp for durable evidence."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
