# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decode one U11 workflow's labeled images, metrics, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from tools.comfy_api import ImageReference, JsonObject
from tools.comfy_integration.history_output import extract_saved_image

from .visual_workflow import BuiltSdxlVisualWorkflow


@dataclass(frozen=True, slots=True)
class SdxlVisualHistoryEvidence:
    """Retain exact terminal values for one labeled visual artifact."""

    image_reference: ImageReference
    metrics: JsonObject
    diagnostics: JsonObject


def decode_sdxl_visual_history(
    history: JsonObject,
    workflow: BuiltSdxlVisualWorkflow,
) -> dict[str, SdxlVisualHistoryEvidence]:
    """Decode every declared terminal output or fail on incomplete evidence."""

    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("U11 history is missing outputs.")
    return {
        output.artifact_id: SdxlVisualHistoryEvidence(
            extract_saved_image(history, output.outputs.save_node_id),
            _single_object(
                outputs,
                output.outputs.metrics_node_id,
                "benchmark_metrics",
                label=output.artifact_id,
            ),
            _single_object(
                outputs,
                output.outputs.diagnostics_node_id,
                "regional_diagnostics",
                label=output.artifact_id,
            ),
        )
        for output in workflow.outputs
    }


def _single_object(
    outputs: dict[object, object],
    node_id: str,
    field: str,
    *,
    label: str,
) -> JsonObject:
    """Return one exact UI object from a terminal output node."""

    output = outputs.get(node_id)
    if not isinstance(output, dict):
        raise ValueError(f"U11 {label} history is missing node {node_id}.")
    values = output.get(field)
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
    ):
        raise ValueError(f"U11 {label} history has invalid {field} evidence.")
    return cast(JsonObject, values[0])
