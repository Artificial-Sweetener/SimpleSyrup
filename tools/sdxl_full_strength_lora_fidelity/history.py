# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decode one full-strength fidelity workflow's terminal evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from tools.comfy_api import ImageReference, JsonObject
from tools.comfy_integration.history_output import extract_saved_image

from .workflow import BuiltFullStrengthFidelityWorkflow


@dataclass(frozen=True, slots=True)
class FullStrengthFidelityEvidence:
    """Retain one image plus exact runtime evidence."""

    image_reference: ImageReference
    metrics: JsonObject
    diagnostics: JsonObject | None


def decode_fidelity_history(
    history: JsonObject,
    workflow: BuiltFullStrengthFidelityWorkflow,
) -> FullStrengthFidelityEvidence:
    """Decode image, metrics, and optional regional diagnostics."""

    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Full-strength fidelity history is missing outputs.")
    diagnostics = None
    if workflow.diagnostics_node_id is not None:
        diagnostics = _single_object(
            outputs,
            workflow.diagnostics_node_id,
            "regional_diagnostics",
        )
    return FullStrengthFidelityEvidence(
        extract_saved_image(history, workflow.save_node_id),
        _single_object(outputs, workflow.metrics_node_id, "benchmark_metrics"),
        diagnostics,
    )


def _single_object(
    outputs: dict[object, object], node_id: str, field: str
) -> JsonObject:
    """Return one exact UI object from a terminal node."""

    output = outputs.get(node_id)
    if not isinstance(output, dict):
        raise ValueError(f"Fidelity history is missing node {node_id}.")
    values = output.get(field)
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
    ):
        raise ValueError(f"Fidelity history has invalid {field} evidence.")
    return cast(JsonObject, values[0])
