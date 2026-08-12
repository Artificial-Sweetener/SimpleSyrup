"""Persist labeled P9.7 images, rejections, logs, and terminal results."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from tools.comfy_api import JsonObject

from .case_artifacts import RegionalPatchInteropCaseArtifactWriter
from .history import RegionalPatchInteropHistory
from .image_evidence import validate_exact_image_equivalence
from .log_evidence import record_regional_patch_interop_logs
from .matrix import RegionalPatchInteropCase
from .source_identity import RepositoryRevision
from .validation import validate_case
from .workflow import BuiltRegionalPatchInteropWorkflow


class RegionalPatchInteropResultRecorder:
    """Own P9.7 validation, durable artifacts, and completion gating."""

    def __init__(
        self,
        root: Path,
        *,
        repositories: tuple[RepositoryRevision, ...],
        model_inventory: tuple[JsonObject, ...],
    ) -> None:
        """Retain exact source and model identities for one managed run."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("P9.7 result root must already exist.")
        if not repositories or not model_inventory:
            raise ValueError("P9.7 results require repository and model identities.")
        self._repositories = repositories
        self._model_inventory = model_inventory
        self._started_at = _utc_now()
        self._observations: list[JsonObject] = []
        self._accepted_images: dict[str, Path] = {}
        self._case_artifacts = RegionalPatchInteropCaseArtifactWriter(self._root)
        self._node_metadata_file: str | None = None
        self._mask_evidence_file: str | None = None
        self._journal = self._root / "p9.7-result.inprogress.json"
        self._result = self._root / "p9.7-result.json"

    def record_node_metadata(self, metadata: dict[str, JsonObject]) -> Path:
        """Persist exact live metadata for every public modifier and sampler."""

        if not metadata:
            raise ValueError("P9.7 node metadata must not be empty.")
        path = self._root / "node-metadata.json"
        self._write_json(path, metadata)
        self._node_metadata_file = path.name
        return path

    def record_masks(
        self,
        groups: dict[str, tuple[str, ...]],
        *,
        input_root: Path,
    ) -> Path:
        """Copy exact submitted full and 1.5x masks with content hashes."""

        destination = self._root / "masks"
        destination.mkdir(exist_ok=False)
        evidence: JsonObject = {}
        for label, names in groups.items():
            records: list[object] = []
            for name in names:
                source = (input_root / name).resolve()
                target = destination / name
                shutil.copyfile(source, target)
                records.append(
                    {
                        "file": str(Path("masks") / name),
                        "sha256": _sha256(target.read_bytes()),
                    }
                )
            evidence[label] = records
        path = self._root / "mask-evidence.json"
        self._write_json(path, evidence)
        self._mask_evidence_file = path.name
        return path

    def record_case(
        self,
        case: RegionalPatchInteropCase,
        workflow: BuiltRegionalPatchInteropWorkflow,
        observed: RegionalPatchInteropHistory,
        *,
        history: JsonObject,
        prompt_id: str,
        wall_runtime_ms: float,
        image_bytes: bytes | None,
        source_image_bytes: bytes | None,
    ) -> Path:
        """Validate and journal one labeled image or rejection sidecar."""

        if any(item.get("case_id") == case.case_id for item in self._observations):
            raise ValueError(f"P9.7 case was already recorded: {case.case_id}.")
        if wall_runtime_ms <= 0.0:
            raise ValueError("P9.7 wall runtime must be positive.")
        history_path = self._root / f"{case.case_id}.history.json"
        workflow_path = self._root / f"{case.case_id}.workflow.json"
        snapshot_path = self._root / f"{case.case_id}.modifier.json"
        self._write_json(history_path, history)
        self._write_json(workflow_path, workflow.prompt)
        self._write_json(snapshot_path, observed.modifier_snapshot)
        validated = validate_case(case, workflow, observed)
        observation: JsonObject = {
            "case_id": case.case_id,
            "label": case.label,
            "model_family": case.model_family.value,
            "spatial_mode": case.spatial_mode.value,
            "modifier": case.modifier.value,
            "scheduled_regional_lora": case.scheduled_regional_lora,
            "status": validated.status,
            "prompt_id": prompt_id,
            "wall_runtime_ms": wall_runtime_ms,
            "model_call_count": validated.model_call_count,
            "diagnostic_record_count": validated.diagnostic_record_count,
            "diagnostic_spatial_modes": list(validated.spatial_modes),
            "workflow_file": workflow_path.name,
            "workflow_sha256": _sha256(workflow_path.read_bytes()),
            "history_file": history_path.name,
            "history_sha256": _sha256(history_path.read_bytes()),
            "modifier_snapshot_file": snapshot_path.name,
            "modifier_snapshot_sha256": _sha256(snapshot_path.read_bytes()),
        }
        artifact_path = self._case_artifacts.write(
            case,
            observed,
            observation,
            image_bytes=image_bytes,
            source_image_bytes=source_image_bytes,
        )
        if case.expect_success:
            self._accepted_images[case.case_id] = artifact_path
        self._observations.append(observation)
        self._persist(self._journal, status="running", completed_at=None)
        return artifact_path

    def finalize(
        self,
        cases: tuple[RegionalPatchInteropCase, ...],
        *,
        system_stats: JsonObject,
        cleanup_verified: bool,
        checkpoint_cleanup_verified: bool,
    ) -> Path:
        """Publish terminal completion after matrix, logs, and cleanup proof."""

        if tuple(item.get("case_id") for item in self._observations) != tuple(
            case.case_id for case in cases
        ):
            raise ValueError("P9.7 result matrix is incomplete or out of order.")
        if self._node_metadata_file is None or self._mask_evidence_file is None:
            raise ValueError("P9.7 metadata or mask evidence is incomplete.")
        if not cleanup_verified or not checkpoint_cleanup_verified:
            raise ValueError(
                "P9.7 managed process, port, mask, or link cleanup failed."
            )
        image_equivalence = [
            asdict(item)
            for item in validate_exact_image_equivalence(self._accepted_images)
        ]
        log_evidence = record_regional_patch_interop_logs(self._root)
        self._persist(
            self._result,
            status="completed",
            completed_at=_utc_now(),
            system_stats=system_stats,
            log_evidence=log_evidence,
            image_equivalence=image_equivalence,
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
        log_evidence: JsonObject | None = None,
        image_equivalence: list[dict[str, object]] | None = None,
    ) -> None:
        """Atomically persist the current ordered evidence state."""

        payload = {
            "schema_version": 1,
            "phase": "p9.7",
            "status": status,
            "started_at_utc": self._started_at,
            "completed_at_utc": completed_at,
            "repositories": [asdict(item) for item in self._repositories],
            "model_inventory": self._model_inventory,
            "node_metadata_file": self._node_metadata_file,
            "mask_evidence_file": self._mask_evidence_file,
            "system_stats": {} if system_stats is None else system_stats,
            "log_evidence": {} if log_evidence is None else log_evidence,
            "image_equivalence": (
                [] if image_equivalence is None else image_equivalence
            ),
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


def _sha256(value: bytes) -> str:
    """Hash one complete persisted artifact value."""

    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    """Return one durable UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
