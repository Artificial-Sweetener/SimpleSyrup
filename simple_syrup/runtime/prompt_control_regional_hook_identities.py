# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Extract stable regional adapter identities from a Prompt Control graph."""

from __future__ import annotations

import math


def prompt_control_regional_hook_identities(
    expansion: object,
) -> tuple[str, ...]:
    """Return ordered identities from explicit lazy CreateHookLora inputs."""

    if not isinstance(expansion, dict):
        raise TypeError("Prompt Control hook expansion must be a dictionary.")
    identities: list[str] = []
    for node_id, raw_node in expansion.items():
        if not isinstance(node_id, str) or not node_id:
            raise TypeError("Prompt Control expansion node ids must be text.")
        if not isinstance(raw_node, dict):
            raise TypeError(f"Prompt Control expansion node {node_id!r} is invalid.")
        if raw_node.get("class_type") != "CreateHookLora":
            continue
        inputs = raw_node.get("inputs")
        if not isinstance(inputs, dict):
            raise TypeError(f"Prompt Control hook node {node_id!r} lacks inputs.")
        name = inputs.get("lora_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Prompt Control hook node {node_id!r} lacks a LoRA name.")
        model_strength = _finite_number(
            inputs.get("strength_model"),
            node_id=node_id,
            field="strength_model",
        )
        clip_strength = _finite_number(
            inputs.get("strength_clip"),
            node_id=node_id,
            field="strength_clip",
        )
        identities.append(f"pc-{name}-{model_strength}-{clip_strength}")
    if not identities:
        raise ValueError(
            "Prompt Control lazy expansion created no regional LoRA hooks."
        )
    if len(set(identities)) != len(identities):
        raise ValueError("Prompt Control regional LoRA identities must be unique.")
    return tuple(identities)


def _finite_number(value: object, *, node_id: str, field: str) -> float:
    """Return one finite hook strength without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(
            f"Prompt Control hook node {node_id!r} {field} must be numeric."
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"Prompt Control hook node {node_id!r} {field} must be finite."
        )
    return result
