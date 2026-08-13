# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate and persist the complete labeled Phase 8 SDXL evidence."""

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
from .history import SdxlModeHistoryEvidence
from .matrix import (
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_STABLE_NAME,
    MODES,
    REFINEMENT_DENOISE,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    UPSCALE_FACTOR,
)
from .workflow import BuiltSdxlAttentionCouplingWorkflow


class SdxlIntegrationResultRecorder:
    """Own Phase 8 SDXL acceptance validation and durable persistence."""

    def __init__(self, root: Path) -> None:
        """Retain one existing collision-resistant managed-run directory."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("SDXL result root must already exist.")
        self._observations: list[JsonObject] = []

    def record_workflow(
        self,
        workflow: BuiltSdxlAttentionCouplingWorkflow,
        *,
        history: JsonObject,
        prompt_id: str,
        evidence: dict[str, SdxlModeHistoryEvidence],
        image_bytes: dict[str, bytes],
        masks: tuple[dict[str, object], ...],
        wall_runtime_ms: float,
    ) -> None:
        """Validate and persist all mode outputs before external cleanup."""

        if wall_runtime_ms <= 0:
            raise ValueError("SDXL integration wall runtime must be positive.")
        expected_ids = tuple(mode.mode_id for mode in MODES)
        if tuple(evidence) != expected_ids or tuple(image_bytes) != expected_ids:
            raise ValueError("SDXL integration evidence must retain matrix order.")
        _write_json(
            self._root / "workflow.json",
            cast(JsonObject, workflow.prompt),
        )
        _write_json(self._root / "history.json", history)
        mask_payload: JsonObject = {"masks": list(masks)}
        _write_json(self._root / "masks.json", mask_payload)
        for mode in MODES:
            terminal = evidence[mode.mode_id]
            data = image_bytes[mode.mode_id]
            dimensions, dynamic_range = sdxl_image_evidence(data)
            if dimensions != (mode.width, mode.height):
                raise ValueError(
                    f"SDXL {mode.mode_id} image must be {mode.width}x{mode.height}; "
                    f"observed {dimensions}."
                )
            if dynamic_range == 0:
                raise ValueError(f"SDXL {mode.mode_id} image is constant.")
            metrics = validate_sdxl_metrics(terminal.metrics, mode.mode_id)
            diagnostics = validate_sdxl_diagnostics(
                terminal.diagnostics,
                label=mode.mode_id,
                expected_spatial_modes=mode.expected_spatial_modes,
            )
            if metrics["model_call_count"] != mode.expected_model_calls:
                raise ValueError(
                    f"SDXL {mode.mode_id} expected {mode.expected_model_calls} "
                    f"model calls, observed {metrics['model_call_count']}."
                )
            image_path = self._root / f"{mode.mode_id}.png"
            image_path.write_bytes(data)
            self._observations.append(
                {
                    "mode_id": mode.mode_id,
                    "label": mode.label,
                    "public_node_id": mode.node_id,
                    "prompt_id": prompt_id,
                    "image_file": image_path.name,
                    "image_width": dimensions[0],
                    "image_height": dimensions[1],
                    "image_dynamic_range": dynamic_range,
                    "image_size_bytes": len(data),
                    "image_sha256": hashlib.sha256(data).hexdigest(),
                    "image_reference": {
                        "filename": terminal.image_reference.filename,
                        "subfolder": terminal.image_reference.subfolder,
                        "type": terminal.image_reference.output_type,
                    },
                    "metrics": metrics,
                    "regional_diagnostics": diagnostics,
                    "wall_runtime_ms": wall_runtime_ms,
                }
            )
        self._persist(status="workflow_completed")

    def finalize(
        self,
        *,
        system_stats: JsonObject,
        server_cleanup: bool,
        checkpoint_cleanup: bool,
        mask_cleanup: bool,
    ) -> Path:
        """Persist completion only after every exact external owner is clean."""

        if tuple(item["mode_id"] for item in self._observations) != tuple(
            mode.mode_id for mode in MODES
        ):
            raise ValueError("SDXL integration observations are incomplete.")
        if not all((server_cleanup, checkpoint_cleanup, mask_cleanup)):
            raise ValueError("SDXL integration requires exact external cleanup.")
        return self._persist(
            status="completed",
            extra={
                "checkpoint": {
                    "stable_name": CHECKPOINT_STABLE_NAME,
                    "size": CHECKPOINT_SIZE,
                    "sha256": CHECKPOINT_SHA256,
                },
                "source_size": [SOURCE_WIDTH, SOURCE_HEIGHT],
                "target_size": [TARGET_WIDTH, TARGET_HEIGHT],
                "upscale_factor": UPSCALE_FACTOR,
                "refinement_denoise": REFINEMENT_DENOISE,
                "system_stats": system_stats,
                "cleanup": {
                    "server": True,
                    "checkpoint_link": True,
                    "masks": True,
                },
            },
        )

    def _persist(
        self,
        *,
        status: str,
        extra: JsonObject | None = None,
    ) -> Path:
        """Atomically write the current authoritative result."""

        payload: JsonObject = {
            "status": status,
            "observations": self._observations,
        }
        if extra is not None:
            payload.update(extra)
        path = self._root / "result.json"
        _write_json(path, payload)
        return path


def _write_json(path: Path, payload: JsonObject) -> None:
    """Atomically persist one stable JSON object."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
