"""Decode labeled images, metrics, and diagnostics from managed history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from tools.comfy_api import ImageReference, JsonObject
from tools.comfy_integration.history_output import extract_saved_image

from .workflow import BuiltSdxlAttentionCouplingWorkflow


@dataclass(frozen=True, slots=True)
class SdxlModeHistoryEvidence:
    """Retain exact terminal history values for one sampler mode."""

    image_reference: ImageReference
    metrics: JsonObject
    diagnostics: JsonObject


def decode_sdxl_history(
    history: JsonObject,
    workflow: BuiltSdxlAttentionCouplingWorkflow,
) -> dict[str, SdxlModeHistoryEvidence]:
    """Decode all labeled terminal outputs or fail on incomplete evidence."""

    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("SDXL integration history is missing outputs.")
    return {
        mode_id: SdxlModeHistoryEvidence(
            image_reference=extract_saved_image(history, values.save_node_id),
            metrics=_single_object(
                outputs,
                values.metrics_node_id,
                "benchmark_metrics",
                mode_id=mode_id,
            ),
            diagnostics=_single_object(
                outputs,
                values.diagnostics_node_id,
                "regional_diagnostics",
                mode_id=mode_id,
            ),
        )
        for mode_id, values in workflow.outputs.items()
    }


def _single_object(
    outputs: dict[object, object],
    node_id: str,
    field: str,
    *,
    mode_id: str,
) -> JsonObject:
    """Return one exact UI object from a terminal output node."""

    output = outputs.get(node_id)
    if not isinstance(output, dict):
        raise ValueError(f"SDXL {mode_id} history is missing node {node_id}.")
    values = output.get(field)
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
    ):
        raise ValueError(f"SDXL {mode_id} history has invalid {field} evidence.")
    return cast(JsonObject, values[0])
