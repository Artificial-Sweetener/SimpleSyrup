# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decode benchmark-only regional Anima adapter assignments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from simple_syrup.domain.regional_lora_plan import RegionalLoraBranch


@dataclass(frozen=True, slots=True)
class VisualRegionalAdapter:
    """Assign one exact Comfy LoRA file to one region and CFG branch."""

    region_index: int
    branch: RegionalLoraBranch
    lora_name: str
    strength: float


def decode_visual_regional_adapters(value: str) -> tuple[VisualRegionalAdapter, ...]:
    """Decode a bounded ordered JSON adapter list without inferred defaults."""

    try:
        decoded: object = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("Visual regional adapters must be valid JSON.") from error
    if not isinstance(decoded, list) or not 1 <= len(decoded) <= 4:
        raise ValueError("Visual regional adapters must contain one to four items.")
    adapters = tuple(_decode_adapter(item, index) for index, item in enumerate(decoded))
    return adapters


def _decode_adapter(value: object, index: int) -> VisualRegionalAdapter:
    """Narrow one adapter record with explicit branch and strength fields."""

    if not isinstance(value, dict) or set(value) != {
        "region_index",
        "branch",
        "lora_name",
        "strength",
    }:
        raise ValueError(
            f"Visual regional adapter {index} must contain exactly region_index, "
            "branch, lora_name, and strength."
        )
    region_index = value["region_index"]
    if isinstance(region_index, bool) or not isinstance(region_index, int):
        raise TypeError(f"Visual regional adapter {index} region_index must be int.")
    if region_index < 0:
        raise ValueError(
            f"Visual regional adapter {index} region_index must be non-negative."
        )
    branch_value = value["branch"]
    try:
        branch = RegionalLoraBranch(branch_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Visual regional adapter {index} branch must be positive or negative."
        ) from error
    lora_name = value["lora_name"]
    if not isinstance(lora_name, str) or not lora_name.strip():
        raise ValueError(
            f"Visual regional adapter {index} lora_name must be non-empty."
        )
    strength = value["strength"]
    if (
        isinstance(strength, bool)
        or not isinstance(strength, int | float)
        or not math.isfinite(float(strength))
    ):
        raise TypeError(f"Visual regional adapter {index} strength must be finite.")
    return VisualRegionalAdapter(region_index, branch, lora_name, float(strength))
