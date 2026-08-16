# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Insert and decode one benchmark-only indexed SDXL model profile."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from tools.comfy_api import JsonObject


@dataclass(frozen=True, slots=True)
class BuiltIndexedOperatorProfileWorkflow:
    """Expose one copied profiled graph and its inserted node identities."""

    prompt: dict[str, JsonObject]
    profile_model_node_id: str
    profile_node_id: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every live node type required by the prompt."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


def build_indexed_operator_profile_workflow(
    *,
    prompt: dict[str, JsonObject],
    sampler_node_id: str,
    run_id: str,
    trace_path: Path,
    call_index: int,
) -> BuiltIndexedOperatorProfileWorkflow:
    """Copy one graph and profile the sampler's current model reference."""

    if not run_id.strip():
        raise ValueError("Indexed operator profile run id must be non-empty.")
    if (
        isinstance(call_index, bool)
        or not isinstance(call_index, int)
        or call_index < 1
    ):
        raise ValueError("Indexed operator profile call index must be positive.")
    copied = copy.deepcopy(prompt)
    sampler_inputs = _node_inputs(copied, sampler_node_id)
    model = sampler_inputs.get("model")
    if (
        not isinstance(model, list)
        or len(model) != 2
        or not isinstance(model[0], str)
        or isinstance(model[1], bool)
        or not isinstance(model[1], int)
    ):
        raise ValueError("Profiled sampler model reference is invalid.")
    profile_model_id = "operator-profile-model"
    profile_result_id = "operator-profile-result"
    if profile_model_id in copied or profile_result_id in copied:
        raise ValueError("Profiled graph already contains reserved profile nodes.")
    copied[profile_model_id] = {
        "class_type": "SimpleSyrupBenchmark.ProfileIndexedModelCall",
        "inputs": {
            "model": model,
            "run_id": run_id,
            "trace_path": str(trace_path.resolve()),
            "call_index": call_index,
        },
    }
    sampler_inputs["model"] = [profile_model_id, 0]
    copied[profile_result_id] = {
        "class_type": "SimpleSyrupBenchmark.ReadOperatorProfile",
        "inputs": {
            "latent": [sampler_node_id, 0],
            "run_id": run_id,
        },
    }
    return BuiltIndexedOperatorProfileWorkflow(
        prompt=copied,
        profile_model_node_id=profile_model_id,
        profile_node_id=profile_result_id,
    )


def decode_indexed_operator_profile(
    history: JsonObject,
    node_id: str,
) -> JsonObject:
    """Decode one complete profile object from its benchmark terminal."""

    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Operator profile history is missing outputs.")
    output = outputs.get(node_id)
    if not isinstance(output, dict):
        raise ValueError("Operator profile history is missing its terminal.")
    values = output.get("operator_profile")
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError("Operator profile terminal must expose one capture.")
    capture = values[0]
    if not isinstance(capture, dict) or not all(
        isinstance(key, str) for key in capture
    ):
        raise TypeError("Operator profile capture must be a JSON object.")
    return cast(JsonObject, capture)


def _node_inputs(
    prompt: dict[str, JsonObject],
    node_id: str,
) -> dict[str, object]:
    """Return one required node's dynamic input object."""

    node = prompt.get(node_id)
    if not isinstance(node, dict):
        raise ValueError("Profiled graph is missing its sampler node.")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Profiled sampler inputs are invalid.")
    return inputs
