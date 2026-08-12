"""Persist ordered labeled P9.6 non-diffusion rejection artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from tools.anima_regional_lora_admission_integration.history import (
    RegionalLoraAdmissionHistory,
)
from tools.anima_regional_lora_admission_integration.workflow import (
    BuiltRegionalLoraAdmissionWorkflow,
)
from tools.comfy_api import JsonObject

from .matrix import (
    PINNED_TURBO,
    AnimaNondiffusionRejectionCase,
    PinnedTurboArtifactIdentity,
)
from .validation import validate_rejection


class AnimaNondiffusionRejectionResultRecorder:
    """Own P9.6 sidecars, hashes, cleanup gating, and terminal publication."""

    def __init__(self, root: Path) -> None:
        """Retain one existing managed artifact root."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("P9.6 result root must already exist.")
        self._started_at = _utc_now()
        self._observations: list[JsonObject] = []
        self._journal = self._root / "p9.6-result.inprogress.json"
        self._result = self._root / "p9.6-result.json"

    def record_metadata(
        self,
        public_sampler: JsonObject,
        fixture_node: JsonObject,
    ) -> Path:
        """Persist exact live product and benchmark-only node metadata."""

        path = self._root / "node-metadata.json"
        self._write_json(
            path,
            {"public_sampler": public_sampler, "fixture_node": fixture_node},
        )
        return path

    def record_masks(self, names: tuple[str, ...], *, input_root: Path) -> Path:
        """Copy exact submitted masks and retain their content digests."""

        destination = self._root / "masks"
        destination.mkdir(exist_ok=False)
        entries: list[object] = []
        for name in names:
            source = (input_root / name).resolve()
            target = destination / name
            shutil.copyfile(source, target)
            entries.append(
                {
                    "file": str(Path("masks") / name),
                    "sha256": _sha256(target.read_bytes()),
                }
            )
        path = self._root / "mask-evidence.json"
        self._write_json(path, {"vertical-hard-50-50": entries})
        return path

    def record_case(
        self,
        case: AnimaNondiffusionRejectionCase,
        workflow: BuiltRegionalLoraAdmissionWorkflow,
        observed: RegionalLoraAdmissionHistory,
        *,
        history: JsonObject,
        prompt_id: str,
        wall_runtime_ms: float,
    ) -> Path:
        """Validate and publish one labeled no-sampling rejection sidecar."""

        if any(item.get("case_id") == case.case_id for item in self._observations):
            raise ValueError(f"P9.6 case was already recorded: {case.case_id}.")
        if wall_runtime_ms <= 0.0:
            raise ValueError("P9.6 wall runtime must be positive.")
        validated = validate_rejection(case, workflow, observed)
        history_path = self._root / f"{case.case_id}.history.json"
        workflow_path = self._root / f"{case.case_id}.workflow.json"
        rejection_path = self._root / f"{case.case_id}.rejection.json"
        self._write_json(history_path, history)
        self._write_json(workflow_path, workflow.workflow.prompt)
        self._write_json(
            rejection_path,
            {
                "case_id": case.case_id,
                "label": case.label,
                "status": "rejected_before_sampling",
                "issue_count": validated.issue_count,
                "exception_type": validated.exception_type,
                "exception_message": validated.exception_message,
            },
        )
        self._observations.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "status": "rejected",
                "prompt_id": prompt_id,
                "wall_runtime_ms": wall_runtime_ms,
                "model_call_count": 0,
                "diagnostic_record_count": 0,
                "image": None,
                "issue_count": validated.issue_count,
                "exception_type": validated.exception_type,
                "exception_message": validated.exception_message,
                "workflow_file": workflow_path.name,
                "workflow_sha256": _sha256(workflow_path.read_bytes()),
                "history_file": history_path.name,
                "history_sha256": _sha256(history_path.read_bytes()),
                "rejection_file": rejection_path.name,
                "rejection_sha256": _sha256(rejection_path.read_bytes()),
            }
        )
        self._persist(self._journal, status="running", completed_at=None)
        return rejection_path

    def finalize(
        self,
        cases: tuple[AnimaNondiffusionRejectionCase, ...],
        *,
        system_stats: JsonObject,
        cleanup_verified: bool,
    ) -> Path:
        """Publish completion only after exact ordered matrix and cleanup."""

        if tuple(item.get("case_id") for item in self._observations) != tuple(
            case.case_id for case in cases
        ):
            raise ValueError("P9.6 result matrix is incomplete or out of order.")
        if not cleanup_verified:
            raise ValueError("P9.6 managed process, port, or mask cleanup failed.")
        self._persist(
            self._result,
            status="completed",
            completed_at=_utc_now(),
            system_stats=system_stats,
        )
        self._journal.unlink(missing_ok=True)
        return self._result

    def _persist(
        self,
        path: Path,
        *,
        status: str,
        completed_at: str | None,
        system_stats: JsonObject | None = None,
    ) -> None:
        """Atomically persist evidence in authoritative matrix order."""

        payload = {
            "schema_version": 1,
            "phase": "p9.6",
            "status": status,
            "started_at_utc": self._started_at,
            "completed_at_utc": completed_at,
            "turbo_artifact": _turbo_payload(PINNED_TURBO),
            "system_stats": {} if system_stats is None else system_stats,
            "observations": self._observations,
        }
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        """Write one stable UTF-8 JSON sidecar."""

        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _turbo_payload(identity: PinnedTurboArtifactIdentity) -> JsonObject:
    """Return public stable Turbo identity without a local model path."""

    return {
        "stable_name": identity.stable_name,
        "lora_name": identity.lora_name,
        "size_bytes": identity.size_bytes,
        "sha256": identity.sha256,
    }


def _sha256(value: bytes) -> str:
    """Hash one persisted artifact value."""

    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    """Return one durable UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
