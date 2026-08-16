# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decode exact metrics and regional diagnostics from image-free history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from tools.comfy_api import JsonObject


@dataclass(frozen=True, slots=True)
class SdxlVisualDiagnosticEvidence:
    """Retain one complete metrics and diagnostics terminal pair."""

    metrics: JsonObject
    diagnostics: JsonObject


def decode_sdxl_visual_diagnostic_history(
    history: JsonObject,
    *,
    metrics_node_id: str,
    diagnostics_node_id: str,
) -> SdxlVisualDiagnosticEvidence:
    """Return both terminal objects or fail closed on incomplete history."""

    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("SDXL visual diagnostic history is missing outputs.")
    return SdxlVisualDiagnosticEvidence(
        _single_object(outputs, metrics_node_id, "benchmark_metrics"),
        _single_object(outputs, diagnostics_node_id, "regional_diagnostics"),
    )


def _single_object(
    outputs: dict[object, object],
    node_id: str,
    field: str,
) -> JsonObject:
    """Return one exact UI object from a required terminal node."""

    output = outputs.get(node_id)
    if not isinstance(output, dict):
        raise ValueError(f"SDXL visual diagnostic history is missing node {node_id}.")
    values = output.get(field)
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
    ):
        raise ValueError(f"SDXL visual diagnostic history has invalid {field}.")
    return cast(JsonObject, values[0])
