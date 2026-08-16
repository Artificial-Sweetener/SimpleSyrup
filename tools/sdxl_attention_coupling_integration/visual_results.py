# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate and incrementally persist the complete U11 SDXL visual evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from tools.comfy_api import JsonObject

from .evidence_validation import (
    sdxl_image_evidence,
    validate_sdxl_diagnostics,
    validate_sdxl_metrics,
)
from .matrix import MODES, SdxlIntegrationMode
from .sampling_controls import SDXL_VISUAL_SAMPLING
from .visual_case_model import RegionalVisualAdapter, SdxlVisualCase, VisualMode
from .visual_history import SdxlVisualHistoryEvidence
from .visual_runtime_expectations import SDXL_VISUAL_RUNTIME_EXPECTATIONS
from .visual_workflow import BuiltSdxlVisualWorkflow


class SdxlVisualResultRecorder:
    """Own validated, durable, case-by-case U11 result persistence."""

    def __init__(self, root: Path, *, cases: tuple[SdxlVisualCase, ...]) -> None:
        """Retain one existing collision-resistant managed-run directory."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("U11 result root must already exist.")
        if not isinstance(cases, tuple) or not cases:
            raise ValueError("U11 result recorder requires declared cases.")
        self._cases = cases
        self._observations: list[JsonObject] = []
        self._failures: list[JsonObject] = []
        self._persist("starting")

    def record_case(
        self,
        workflow: BuiltSdxlVisualWorkflow,
        *,
        case: SdxlVisualCase,
        history: JsonObject,
        prompt_id: str,
        evidence: dict[str, SdxlVisualHistoryEvidence],
        image_bytes: dict[str, bytes],
        wall_runtime_ms: float,
    ) -> None:
        """Validate and persist one completed case before the next submission."""

        expected_ids = tuple(output.artifact_id for output in workflow.outputs)
        if tuple(evidence) != expected_ids or tuple(image_bytes) != expected_ids:
            raise ValueError("U11 case evidence must retain declared output order.")
        if wall_runtime_ms <= 0:
            raise ValueError("U11 case wall runtime must be positive.")
        case_root = self._root / case.case_id
        case_root.mkdir(exist_ok=False)
        _write_json(case_root / "workflow.json", cast(JsonObject, workflow.prompt))
        _write_json(case_root / "history.json", history)
        for output in workflow.outputs:
            terminal = evidence[output.artifact_id]
            data = image_bytes[output.artifact_id]
            mode = _mode(output.mode)
            dimensions, dynamic_range = sdxl_image_evidence(data)
            if dimensions != (mode.width, mode.height):
                raise ValueError(
                    f"U11 {output.artifact_id} expected {mode.width}x{mode.height}, "
                    f"observed {dimensions}."
                )
            if dynamic_range == 0:
                raise ValueError(f"U11 {output.artifact_id} image is constant.")
            metrics = validate_sdxl_metrics(terminal.metrics, output.artifact_id)
            diagnostics = validate_sdxl_diagnostics(
                terminal.diagnostics,
                label=output.artifact_id,
                expected_spatial_modes=mode.expected_spatial_modes,
            )
            expected_model_calls = (
                SDXL_VISUAL_RUNTIME_EXPECTATIONS.expected_model_calls(case, mode)
            )
            if metrics["model_call_count"] != expected_model_calls:
                raise ValueError(
                    f"U11 {output.artifact_id} expected {expected_model_calls} "
                    f"model calls, observed {metrics['model_call_count']}."
                )
            image_path = case_root / f"{output.mode.value}.png"
            image_path.write_bytes(data)
            self._observations.append(
                {
                    "artifact_id": output.artifact_id,
                    "case_id": case.case_id,
                    "label": case.label,
                    "mode": output.mode.value,
                    "seed": SDXL_VISUAL_SAMPLING.seed,
                    "mask_profile": case.mask_profile.value,
                    "regional_prompt_start_percent": (
                        case.regional_prompt_start_percent
                    ),
                    "regional_prompt_weight": case.regional_prompt_weight,
                    "prompts": {
                        "base_positive_g": case.base_positive_g,
                        "base_positive_l": case.base_positive_l,
                        "base_negative_g": case.base_negative_g,
                        "base_negative_l": case.base_negative_l,
                        "left_positive_g": case.left_g,
                        "left_positive_l": case.left_l,
                        "left_negative_g": case.left_negative_g,
                        "left_negative_l": case.left_negative_l,
                        "right_positive_g": case.right_g,
                        "right_positive_l": case.right_l,
                        "right_negative_g": case.right_negative_g,
                        "right_negative_l": case.right_negative_l,
                    },
                    "global_adapters": [
                        {"name": item.lora_name, "strength": item.strength}
                        for item in case.global_adapters
                    ],
                    "left_adapters": [
                        _regional_json(item) for item in case.left_adapters
                    ],
                    "right_adapters": [
                        _regional_json(item) for item in case.right_adapters
                    ],
                    "prompt_id": prompt_id,
                    "image_file": str(image_path.relative_to(self._root)),
                    "image_width": dimensions[0],
                    "image_height": dimensions[1],
                    "image_dynamic_range": dynamic_range,
                    "image_size_bytes": len(data),
                    "image_sha256": hashlib.sha256(data).hexdigest(),
                    "metrics": metrics,
                    "regional_diagnostics": diagnostics,
                    "case_wall_runtime_ms": wall_runtime_ms,
                }
            )
        self._persist("running")

    def record_failure(self, case: SdxlVisualCase, error: BaseException) -> None:
        """Persist one exact failed case without discarding accepted predecessors."""

        self._failures.append(
            {
                "case_id": case.case_id,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        self._persist("failed")

    def finalize(
        self,
        *,
        system_stats: JsonObject,
        server_cleanup: bool,
        model_cleanup: bool,
        mask_cleanup: bool,
        mask_evidence: tuple[dict[str, object], ...],
    ) -> Path:
        """Complete only after every declared output and external owner are clean."""

        expected = tuple(
            f"{case.case_id}--{mode.value}"
            for case in self._cases
            for mode in case.modes
        )
        observed = tuple(cast(str, item["artifact_id"]) for item in self._observations)
        if observed != expected or self._failures:
            raise ValueError("U11 visual observations are incomplete or failed.")
        if not all((server_cleanup, model_cleanup, mask_cleanup)):
            raise ValueError("U11 visual integration requires exact external cleanup.")
        return self._persist(
            "completed",
            extra={
                "system_stats": system_stats,
                "mask_evidence": list(mask_evidence),
                "cleanup": {"server": True, "model_links": True, "masks": True},
            },
        )

    def _persist(self, status: str, *, extra: JsonObject | None = None) -> Path:
        """Atomically replace the current authoritative U11 result."""

        payload: JsonObject = {
            "status": status,
            "observations": self._observations,
            "failures": self._failures,
        }
        if extra is not None:
            payload.update(extra)
        path = self._root / "u11-result.json"
        _write_json(path, payload)
        return path


def _mode(mode: VisualMode) -> SdxlIntegrationMode:
    """Return the existing exact mode contract for one visual mode."""

    return next(item for item in MODES if item.mode_id == mode.value)


def _regional_json(adapter: RegionalVisualAdapter) -> JsonObject:
    """Serialize one narrowed regional adapter declaration."""
    return {
        "name": adapter.lora_name,
        "model_strength": adapter.model_strength,
        "clip_strength": adapter.clip_strength,
        "schedule": [list(boundary) for boundary in adapter.schedule],
    }


def _write_json(path: Path, payload: JsonObject) -> None:
    """Atomically persist one stable JSON object."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
