# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decode Prompt Control snapshot and runtime evidence from Comfy history."""

from __future__ import annotations

from dataclasses import dataclass

from tools.comfy_api import JsonObject


@dataclass(frozen=True)
class PromptControlOutputs:
    """Hold one normalized pre-sampling snapshot and runtime observation."""

    expansion: JsonObject
    snapshot: JsonObject
    runtime: JsonObject


def parse_outputs(
    history: JsonObject,
    *,
    expansion_node_id: str,
    snapshot_node_id: str,
    runtime_node_id: str,
) -> PromptControlOutputs:
    """Narrow one successful history record into P0.8 evidence."""

    status = _object(history.get("status"), "history.status")
    if status.get("status_str") != "success":
        raise RuntimeError(f"ComfyUI execution failed: {status.get('messages')!r}.")
    outputs = _object(history.get("outputs"), "history.outputs")
    expansion_output = _object(outputs.get(expansion_node_id), "expansion output")
    snapshot_output = _object(outputs.get(snapshot_node_id), "snapshot output")
    runtime_output = _object(outputs.get(runtime_node_id), "runtime output")
    expansion = _single_object(
        expansion_output.get("prompt_control_expansion"), "prompt_control_expansion"
    )
    snapshot = _single_object(
        snapshot_output.get("prompt_control_snapshot"), "prompt_control_snapshot"
    )
    runtime = _single_object(
        runtime_output.get("prompt_control_runtime"), "prompt_control_runtime"
    )
    return PromptControlOutputs(expansion, snapshot, runtime)


def _single_object(value: object, field: str) -> JsonObject:
    """Read one JSON object from an output list."""

    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{field} must contain exactly one record.")
    return _object(value[0], field)


def _object(value: object, field: str) -> JsonObject:
    """Narrow a JSON object with string keys."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} must be a JSON object.")
    return value
