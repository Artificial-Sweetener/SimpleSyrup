# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persist the bounded full-strength LoRA fidelity comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from PIL import Image

from tools.comfy_api import JsonObject

from .cases import FullStrengthFidelityCase
from .history import FullStrengthFidelityEvidence
from .workflow import BuiltFullStrengthFidelityWorkflow


class FullStrengthFidelityResultRecorder:
    """Own case-by-case evidence and exact completion validation."""

    def __init__(
        self, root: Path, *, cases: tuple[FullStrengthFidelityCase, ...]
    ) -> None:
        """Retain one existing managed run and its ordered cases."""

        self._root = root.resolve()
        if not self._root.is_dir() or not cases:
            raise ValueError("Fidelity recorder requires a run root and cases.")
        self._cases = cases
        self._observations: list[JsonObject] = []
        self._failures: list[JsonObject] = []
        self._persist("starting")

    def record_case(
        self,
        workflow: BuiltFullStrengthFidelityWorkflow,
        *,
        case: FullStrengthFidelityCase,
        history: JsonObject,
        prompt_id: str,
        evidence: FullStrengthFidelityEvidence,
        image_bytes: bytes,
        wall_runtime_ms: float,
    ) -> Path:
        """Persist one labeled original-resolution artifact before continuing."""

        case_root = self._root / case.case_id
        case_root.mkdir(exist_ok=False)
        _write_json(case_root / "workflow.json", cast(JsonObject, workflow.prompt))
        _write_json(case_root / "history.json", history)
        image_path = case_root / "full.png"
        image_path.write_bytes(image_bytes)
        with Image.open(image_path) as image:
            dimensions = image.size
        if dimensions != (1024, 1024):
            raise ValueError(f"Fidelity image has unexpected dimensions: {dimensions}.")
        self._observations.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "execution_mode": case.mode.value,
                "character_id": case.character.character_id,
                "model_strength": case.model_strength,
                "clip_strength": case.clip_strength,
                "schedule": [list(boundary) for boundary in case.schedule],
                "seed": 7429113057,
                "steps": 30,
                "sampler": "euler_ancestral",
                "scheduler": "karras",
                "prompt_id": prompt_id,
                "image_file": str(image_path.relative_to(self._root)),
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "metrics": evidence.metrics,
                "regional_diagnostics": evidence.diagnostics,
                "wall_runtime_ms": wall_runtime_ms,
            }
        )
        self._persist("running")
        return image_path

    def record_failure(
        self, case: FullStrengthFidelityCase, error: BaseException
    ) -> None:
        """Durably retain one failed case and all preceding evidence."""

        self._failures.append(
            {"case_id": case.case_id, "error": f"{type(error).__name__}: {error}"}
        )
        self._persist("failed")

    def finalize(
        self,
        *,
        system_stats: JsonObject,
        server_cleanup: bool,
        model_cleanup: bool,
        mask_cleanup: bool,
        mask_evidence: dict[str, object],
    ) -> Path:
        """Complete only when all four declared artifacts and owners are clean."""

        observed = tuple(cast(str, item["case_id"]) for item in self._observations)
        expected = tuple(case.case_id for case in self._cases)
        if observed != expected or self._failures:
            raise ValueError("Full-strength fidelity evidence is incomplete.")
        if not all((server_cleanup, model_cleanup, mask_cleanup)):
            raise ValueError("Full-strength fidelity external cleanup failed.")
        return self._persist(
            "completed",
            extra={
                "system_stats": system_stats,
                "mask_evidence": mask_evidence,
                "cleanup": {"server": True, "model_links": True, "mask": True},
            },
        )

    def _persist(self, status: str, *, extra: JsonObject | None = None) -> Path:
        """Atomically replace the authoritative result record."""

        payload: JsonObject = {
            "status": status,
            "observations": self._observations,
            "failures": self._failures,
        }
        if extra is not None:
            payload.update(extra)
        path = self._root / "full-strength-fidelity-result.json"
        _write_json(path, payload)
        return path


def _write_json(path: Path, payload: JsonObject) -> None:
    """Atomically write one stable JSON object."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
