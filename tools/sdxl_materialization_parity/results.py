# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decode one managed materialization parity terminal fail-closed."""

from __future__ import annotations

from tools.comfy_api import JsonObject


def decode_materialization_parity(
    history: JsonObject,
    *,
    terminal_node_id: str,
    expected_run_id: str,
) -> JsonObject:
    """Return the one complete validated UI comparison object."""

    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Materialization parity history is missing outputs.")
    output = outputs.get(terminal_node_id)
    if not isinstance(output, dict):
        raise ValueError("Materialization parity history is missing its terminal.")
    values = output.get("materialization_parity")
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError("Materialization parity terminal requires one result.")
    payload = values[0]
    if not isinstance(payload, dict):
        raise ValueError("Materialization parity result must be an object.")
    if payload.get("run_id") != expected_run_id:
        raise ValueError("Materialization parity run identity does not match.")
    _require_non_negative_int(payload, "variant_count")
    _require_non_negative_int(payload, "element_count")
    _require_non_negative_int(payload, "differing_element_count")
    _require_non_negative_int(payload, "peak_vram_bytes")
    if payload["variant_count"] < 1 or payload["element_count"] < 1:
        raise ValueError("Materialization parity result must compare nonempty banks.")
    if payload["differing_element_count"] > payload["element_count"]:
        raise ValueError("Materialization parity differing count exceeds its total.")
    if not isinstance(payload.get("exact"), bool):
        raise ValueError("Materialization parity exactness flag is invalid.")
    variants = payload.get("variants")
    if not isinstance(variants, list) or len(variants) != payload["variant_count"]:
        raise ValueError("Materialization parity variant evidence is incomplete.")
    return {str(key): value for key, value in payload.items()}


def _require_non_negative_int(payload: dict[object, object], key: str) -> None:
    """Validate one required non-negative integer result field."""

    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Materialization parity field {key!r} is invalid.")
