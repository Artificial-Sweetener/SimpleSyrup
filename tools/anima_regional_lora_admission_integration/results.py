# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persist complete labeled P9.5 managed admission evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import cast

from PIL import Image

from tools.comfy_api import ImageReference, JsonObject

from .graph_contract import HEIGHT, WIDTH
from .history import RegionalLoraAdmissionHistory, RegionalLoraAdmissionSuccess
from .matrix import RegionalLoraAdmissionCase
from .validation import validate_history
from .workflow import BuiltRegionalLoraAdmissionWorkflow


class RegionalLoraAdmissionResultRecorder:
    """Own ordered P9.5 sidecars, artifacts, and terminal publication."""

    def __init__(self, root: Path) -> None:
        """Retain one existing managed artifact root."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("P9.5 result root must already exist.")
        self._started_at = _utc_now()
        self._observations: list[JsonObject] = []
        self._journal = self._root / "p9.5-result.inprogress.json"
        self._result = self._root / "p9.5-result.json"

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

    def record_masks(
        self,
        names: tuple[str, ...],
        *,
        input_root: Path,
    ) -> Path:
        """Copy exact submitted masks and retain their content digests."""

        destination = self._root / "masks"
        destination.mkdir(exist_ok=False)
        entries: list[object] = []
        for name in names:
            source = (input_root / name).resolve()
            target = destination / name
            shutil.copyfile(source, target)
            data = target.read_bytes()
            entries.append(
                {
                    "file": str(Path("masks") / name),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        path = self._root / "mask-evidence.json"
        self._write_json(path, {"vertical-hard-50-50": entries})
        return path

    def record_case(
        self,
        case: RegionalLoraAdmissionCase,
        workflow: BuiltRegionalLoraAdmissionWorkflow,
        observed: RegionalLoraAdmissionHistory,
        *,
        history: JsonObject,
        prompt_id: str,
        wall_runtime_ms: float,
        image_reference: ImageReference | None = None,
        image_bytes: bytes | None = None,
    ) -> Path | None:
        """Validate through the policy owner and journal one exact case."""

        if any(item.get("case_id") == case.case_id for item in self._observations):
            raise ValueError(f"P9.5 case was already recorded: {case.case_id}.")
        if wall_runtime_ms <= 0.0:
            raise ValueError("P9.5 wall runtime must be positive.")
        validated = validate_history(case, workflow, observed)
        history_path = self._root / f"{case.case_id}.history.json"
        workflow_path = self._root / f"{case.case_id}.workflow.json"
        self._write_json(history_path, history)
        self._write_json(workflow_path, workflow.workflow.prompt)

        image_path: Path | None = None
        image_evidence: JsonObject | None = None
        if case.expect_success:
            if image_bytes is None or image_reference is None:
                raise ValueError("P9.5 supported case requires one image artifact.")
            self._validate_image(image_bytes)
            image_path = self._root / f"{case.case_id}.png"
            image_path.write_bytes(image_bytes)
            image_evidence = {
                "file": image_path.name,
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "size_bytes": len(image_bytes),
                "reference": {
                    "filename": image_reference.filename,
                    "subfolder": image_reference.subfolder,
                    "type": image_reference.output_type,
                },
            }
        elif image_bytes is not None or image_reference is not None:
            raise ValueError("P9.5 rejected case must not publish an image.")

        metrics = (
            observed.metrics
            if isinstance(observed, RegionalLoraAdmissionSuccess)
            else None
        )
        self._observations.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "status": validated.status,
                "prompt_id": prompt_id,
                "wall_runtime_ms": wall_runtime_ms,
                "model_call_count": validated.model_call_count,
                "diagnostic_record_count": validated.diagnostic_record_count,
                "target_count": validated.target_count,
                "metrics": metrics,
                "exception_type": validated.exception_type,
                "exception_message": validated.exception_message,
                "image": image_evidence,
                "workflow_file": workflow_path.name,
                "history_file": history_path.name,
            }
        )
        self._persist(self._journal, status="running", completed_at=None)
        return image_path

    def finalize(
        self,
        cases: tuple[RegionalLoraAdmissionCase, ...],
        *,
        system_stats: JsonObject,
        cleanup_verified: bool,
    ) -> Path:
        """Publish completion only after exact ordered matrix and cleanup."""

        if tuple(item.get("case_id") for item in self._observations) != tuple(
            case.case_id for case in cases
        ):
            raise ValueError("P9.5 result matrix is incomplete or out of order.")
        if not cleanup_verified:
            raise ValueError("P9.5 managed process, port, or mask cleanup failed.")
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
            "phase": "p9.5",
            "status": status,
            "started_at_utc": self._started_at,
            "completed_at_utc": completed_at,
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
    def _validate_image(image_bytes: bytes) -> None:
        """Require one non-flat exact-size decoded PNG."""

        if not image_bytes:
            raise ValueError("P9.5 image must not be empty.")
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format != "PNG" or image.size != (WIDTH, HEIGHT):
                raise ValueError(f"P9.5 image must be a {WIDTH}x{HEIGHT} PNG.")
            extrema = cast(
                tuple[tuple[int, int], ...], image.convert("RGB").getextrema()
            )
        if not any(high > low for low, high in extrema):
            raise ValueError("P9.5 image must contain non-flat visual output.")

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        """Write one stable UTF-8 JSON sidecar."""

        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _utc_now() -> str:
    """Return one durable UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
