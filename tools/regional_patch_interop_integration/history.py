"""Decode successful or rejected P9.7 terminal Comfy history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from tools.comfy_api import ImageReference, JsonObject
from tools.comfy_integration.history_output import extract_saved_image

from .workflow import BuiltRegionalPatchInteropWorkflow


@dataclass(frozen=True, slots=True)
class RegionalPatchInteropSuccess:
    """Retain accepted image, metric, diagnostic, and modifier evidence."""

    modifier_snapshot: JsonObject
    metrics: JsonObject
    diagnostics: JsonObject
    image_reference: ImageReference
    source_image_reference: ImageReference | None


@dataclass(frozen=True, slots=True)
class RegionalPatchInteropError:
    """Retain exact pre-sampling failure and modifier evidence."""

    modifier_snapshot: JsonObject
    node_id: str
    node_type: str
    exception_type: str
    exception_message: str
    executed_node_ids: tuple[str, ...]


RegionalPatchInteropHistory: TypeAlias = (
    RegionalPatchInteropSuccess | RegionalPatchInteropError
)


def parse_history(
    history: JsonObject,
    workflow: BuiltRegionalPatchInteropWorkflow,
) -> RegionalPatchInteropHistory:
    """Return one exact accepted or rejected terminal evidence value."""

    status = _object(history.get("status"), "history.status")
    outputs = _object(history.get("outputs"), "history.outputs")
    modifier_snapshot = _single(
        outputs,
        workflow.modifier_snapshot_node_id,
        "model_modifier_snapshot",
    )
    status_string = status.get("status_str")
    if status_string == "success":
        if status.get("completed") is not True:
            raise ValueError("P9.7 successful history must be completed.")
        source_reference = (
            None
            if workflow.source_save_node_id is None
            else extract_saved_image(history, workflow.source_save_node_id)
        )
        return RegionalPatchInteropSuccess(
            modifier_snapshot,
            _single(outputs, workflow.metrics_node_id, "benchmark_metrics"),
            _single(
                outputs,
                workflow.diagnostics_node_id,
                "regional_diagnostics",
            ),
            extract_saved_image(history, workflow.save_node_id),
            source_reference,
        )
    if status_string != "error":
        raise ValueError(f"P9.7 history has unsupported status {status_string!r}.")
    for forbidden in (
        workflow.metrics_node_id,
        workflow.diagnostics_node_id,
        workflow.save_node_id,
    ):
        if forbidden in outputs:
            raise ValueError("P9.7 rejected workflow emitted terminal sampler output.")
    error = _execution_error(status.get("messages"))
    return RegionalPatchInteropError(
        modifier_snapshot,
        _string(error.get("node_id"), "execution_error.node_id"),
        _string(error.get("node_type"), "execution_error.node_type"),
        _string(error.get("exception_type"), "execution_error.exception_type"),
        _string(
            error.get("exception_message"),
            "execution_error.exception_message",
        ),
        _string_tuple(error.get("executed"), "execution_error.executed"),
    )


def _execution_error(value: object) -> JsonObject:
    """Return the sole terminal execution-error payload."""

    if not isinstance(value, list):
        raise TypeError("P9.7 history messages must be a list.")
    errors: list[JsonObject] = []
    for message in value:
        if not isinstance(message, list) or len(message) != 2:
            raise TypeError("P9.7 history message entries must be pairs.")
        if message[0] == "execution_error":
            errors.append(_object(message[1], "execution_error"))
    if len(errors) != 1:
        raise ValueError("P9.7 rejected history must contain one execution error.")
    return errors[0]


def _single(outputs: JsonObject, node_id: str, field: str) -> JsonObject:
    """Return one exact UI record from one graph node."""

    output = _object(outputs.get(node_id), f"node {node_id}")
    values = output.get(field)
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"P9.7 {field} must contain exactly one record.")
    return _object(values[0], field)


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    """Return one exact JSON string list."""

    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"P9.7 {field} must be a string list.")
    return tuple(value)


def _string(value: object, field: str) -> str:
    """Return one nonempty JSON string."""

    if not isinstance(value, str) or not value:
        raise TypeError(f"P9.7 {field} must be a nonempty string.")
    return value


def _object(value: object, field: str) -> JsonObject:
    """Return one JSON object with string keys."""

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"P9.7 {field} must be an object.")
    return value
