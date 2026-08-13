# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Narrow successful or rejected regional-LoRA admission history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from tools.comfy_api import JsonObject

from .workflow import BuiltRegionalLoraAdmissionWorkflow


@dataclass(frozen=True, slots=True)
class RegionalLoraAdmissionSuccess:
    """Retain exact metric and diagnostic output records."""

    metrics: JsonObject
    diagnostics: JsonObject


@dataclass(frozen=True, slots=True)
class RegionalLoraAdmissionError:
    """Retain one exact terminal public-sampler execution failure."""

    node_id: str
    node_type: str
    exception_type: str
    exception_message: str
    executed_node_ids: tuple[str, ...]


RegionalLoraAdmissionHistory: TypeAlias = (
    RegionalLoraAdmissionSuccess | RegionalLoraAdmissionError
)


def parse_history(
    history: JsonObject,
    workflow: BuiltRegionalLoraAdmissionWorkflow,
) -> RegionalLoraAdmissionHistory:
    """Return exact success evidence or the one terminal execution error."""

    status = _object(history.get("status"), "history.status")
    status_string = status.get("status_str")
    outputs = _object(history.get("outputs"), "history.outputs")
    if status_string == "success":
        return RegionalLoraAdmissionSuccess(
            _single(
                outputs,
                workflow.workflow.metrics_node_id,
                "benchmark_metrics",
            ),
            _single(
                outputs,
                workflow.workflow.diagnostics_node_id,
                "regional_diagnostics",
            ),
        )
    if status_string != "error":
        raise ValueError(
            f"{_phase(workflow)} history has unsupported status: {status_string!r}."
        )
    for forbidden in (
        workflow.workflow.metrics_node_id,
        workflow.workflow.diagnostics_node_id,
        workflow.workflow.save_node_id,
    ):
        if forbidden in outputs:
            raise ValueError(
                f"{_phase(workflow)} rejected workflow emitted a terminal sampler "
                "output."
            )
    error = _execution_error(status.get("messages"), phase=_phase(workflow))
    return RegionalLoraAdmissionError(
        _string(error.get("node_id"), "execution_error.node_id"),
        _string(error.get("node_type"), "execution_error.node_type"),
        _string(error.get("exception_type"), "execution_error.exception_type"),
        _string(error.get("exception_message"), "execution_error.exception_message"),
        _string_tuple(error.get("executed"), "execution_error.executed"),
    )


def _execution_error(value: object, *, phase: str) -> JsonObject:
    """Return the one terminal execution-error message payload."""

    if not isinstance(value, list):
        raise TypeError(f"{phase} history messages must be a list.")
    errors: list[JsonObject] = []
    for message in value:
        if not isinstance(message, list) or len(message) != 2:
            raise TypeError(f"{phase} history message entries must be pairs.")
        if message[0] == "execution_error":
            errors.append(_object(message[1], "execution_error"))
    if len(errors) != 1:
        raise ValueError(f"{phase} rejected history must contain one execution error.")
    return errors[0]


def _single(outputs: JsonObject, node_id: str, field: str) -> JsonObject:
    """Return one exact UI record from one terminal output node."""

    output = _object(outputs.get(node_id), f"node {node_id}")
    values = output.get(field)
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"Admission {field} must contain exactly one record.")
    return _object(values[0], field)


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    """Narrow one ordered JSON string list."""

    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"Admission {field} must be a string list.")
    return tuple(value)


def _string(value: object, field: str) -> str:
    """Narrow one nonempty JSON string."""

    if not isinstance(value, str) or not value:
        raise TypeError(f"Admission {field} must be a nonempty string.")
    return value


def _object(value: object, field: str) -> JsonObject:
    """Narrow one JSON object with string keys."""

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"Admission {field} must be an object.")
    return value


def _phase(workflow: BuiltRegionalLoraAdmissionWorkflow) -> str:
    """Return the explicit artifact phase in diagnostic display form."""

    return workflow.artifact_phase.upper()
