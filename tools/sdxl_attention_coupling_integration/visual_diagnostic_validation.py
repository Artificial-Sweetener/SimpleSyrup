# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate standard-UNet phase and native self-attention runtime proof."""

from __future__ import annotations

from typing import cast

from tools.comfy_api import JsonObject


def validate_standard_unet_diagnostics(
    diagnostics: JsonObject,
) -> JsonObject:
    """Require complete phases without a standard attn1 replacement."""

    _positive_count(diagnostics, "attention_phase_record_count", "attention-phase")
    phases = _objects(diagnostics, "attention_phases", "attention-phase")
    stages = {value.get("stage") for value in phases}
    required = {"composition", "specialization", "consolidation"}
    if not required <= stages:
        raise ValueError(
            "Standard UNet diagnostics require a complete phase trajectory."
        )
    self_attention_count = diagnostics.get("self_attention_record_count")
    if (
        isinstance(self_attention_count, bool)
        or not isinstance(self_attention_count, int)
        or self_attention_count != 0
    ):
        raise ValueError(
            "Standard UNet diagnostics require zero self-attention records."
        )
    if diagnostics.get("self_attention") != []:
        raise ValueError("Standard UNet diagnostics require no self-attention values.")
    return diagnostics


def _positive_count(diagnostics: JsonObject, field: str, label: str) -> int:
    """Return one required positive diagnostic record count."""

    value = diagnostics.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Standard UNet diagnostics require positive {label} records.")
    return value


def _objects(
    diagnostics: JsonObject,
    field: str,
    label: str,
) -> tuple[JsonObject, ...]:
    """Return one non-empty ordered diagnostic object sequence."""

    values = diagnostics.get(field)
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, dict) for value in values)
    ):
        raise ValueError(f"Standard UNet diagnostics require {label} values.")
    return tuple(cast(JsonObject, value) for value in values)
