# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate and persist P9.3 managed global/regional LoRA evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.anima_attention_coupling_workflow import (
    STEPS,
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.comfy_api import ImageReference, JsonObject

from .image_validation import GLOBAL_REGIONAL_LORA_IMAGE_VALIDATOR
from .matrix import GlobalRegionalLoraCase

_OVERLAP_MESSAGE = "Regional Anima LoRA content is already applied globally"


class GlobalRegionalLoraResultRecorder:
    """Own terminal status, ordering, and durable P9.3 evidence persistence."""

    def __init__(self, root: Path) -> None:
        """Retain one already-created managed artifact directory."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("P9.3 result root must already exist.")
        self._observations: list[JsonObject] = []
        self._image_paths: dict[str, Path] = {}

    def record_success(
        self,
        case: GlobalRegionalLoraCase,
        workflow: BuiltAnimaAttentionCouplingWorkflow,
        *,
        prompt_id: str,
        history: JsonObject,
        reference: ImageReference,
        image_bytes: bytes,
    ) -> Path:
        """Require success and preserve its labeled image and sidecars."""

        if case.expect_overlap_error:
            raise ValueError("P9.3 overlap case cannot be recorded as success.")
        _require_status(history, "success")
        metrics = _metrics(history, workflow)
        image_path = self._root / f"{case.case_id}.png"
        image_path.write_bytes(image_bytes)
        self._image_paths[case.case_id] = image_path
        self._sidecars(case, workflow, history)
        self._observations.append(
            {
                "case_id": case.case_id,
                "label": case.integration.label,
                "status": "success",
                "prompt_id": prompt_id,
                "metrics": metrics,
                "image_file": image_path.name,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "image_reference": {
                    "filename": reference.filename,
                    "subfolder": reference.subfolder,
                    "type": reference.output_type,
                },
            }
        )
        return image_path

    def record_overlap_rejection(
        self,
        case: GlobalRegionalLoraCase,
        workflow: BuiltAnimaAttentionCouplingWorkflow,
        *,
        prompt_id: str,
        history: JsonObject,
    ) -> None:
        """Require the exact pre-sampling duplicate diagnostic and no image."""

        if not case.expect_overlap_error:
            raise ValueError("P9.3 success case cannot be recorded as rejection.")
        _require_status(history, "error")
        serialized = json.dumps(history, sort_keys=True)
        if (
            _OVERLAP_MESSAGE not in serialized
            or "adapter-a.safetensors" not in serialized
        ):
            raise ValueError("P9.3 history lacks the exact overlap diagnostic.")
        if workflow.save_node_id in _outputs(history):
            raise ValueError("P9.3 overlap rejection unexpectedly saved an image.")
        self._sidecars(case, workflow, history)
        self._observations.append(
            {
                "case_id": case.case_id,
                "label": case.integration.label,
                "status": "rejected_before_sampling",
                "prompt_id": prompt_id,
                "diagnostic": _OVERLAP_MESSAGE,
            }
        )

    def finalize(
        self,
        definitions: tuple[GlobalRegionalLoraCase, ...],
        *,
        system_stats: JsonObject,
        cleanup_verified: bool,
        masks_removed: bool,
    ) -> Path:
        """Write completion only after ordered coverage and cleanup proof."""

        expected = tuple(case.case_id for case in definitions)
        observed = tuple(str(item["case_id"]) for item in self._observations)
        if observed != expected:
            raise ValueError("P9.3 observations are incomplete or out of order.")
        if not cleanup_verified or not masks_removed:
            raise ValueError("P9.3 managed process, port, or mask cleanup failed.")
        transition = GLOBAL_REGIONAL_LORA_IMAGE_VALIDATOR.validate(self._image_paths)
        path = self._root / "p9.3-result.json"
        _write_json(
            path,
            {
                "schema_version": 1,
                "phase": "p9.3",
                "status": "completed",
                "observations": self._observations,
                "transition": transition.as_json(),
                "system_stats": system_stats,
                "cleanup": {
                    "process_stopped": True,
                    "port_available": True,
                    "masks_removed": True,
                },
            },
        )
        return path

    def _sidecars(
        self,
        case: GlobalRegionalLoraCase,
        workflow: BuiltAnimaAttentionCouplingWorkflow,
        history: JsonObject,
    ) -> None:
        """Persist exact submitted graph and returned history."""

        _write_json(self._root / f"{case.case_id}.workflow.json", workflow.prompt)
        _write_json(self._root / f"{case.case_id}.history.json", history)


def _require_status(history: JsonObject, expected: str) -> None:
    """Require one exact terminal Comfy status."""

    status = history.get("status")
    if not isinstance(status, dict) or status.get("status_str") != expected:
        raise ValueError(f"P9.3 history must have status {expected!r}.")


def _outputs(history: JsonObject) -> dict[str, object]:
    """Return the exact output mapping or an empty mapping for early failure."""

    outputs = history.get("outputs", {})
    if not isinstance(outputs, dict):
        raise TypeError("P9.3 history outputs must be an object.")
    return outputs


def _metrics(
    history: JsonObject,
    workflow: BuiltAnimaAttentionCouplingWorkflow,
) -> JsonObject:
    """Require one complete denoiser call per configured sampling step."""

    output = _outputs(history).get(workflow.metrics_node_id)
    if not isinstance(output, dict):
        raise ValueError("P9.3 successful history is missing metrics.")
    values = output.get("benchmark_metrics")
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
    ):
        raise ValueError("P9.3 successful history requires one metrics record.")
    metrics = values[0]
    if metrics.get("model_call_count") != STEPS:
        raise ValueError("P9.3 success must retain one model call per step.")
    return metrics


def _write_json(path: Path, value: object) -> None:
    """Write deterministic human-reviewable JSON evidence."""

    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
