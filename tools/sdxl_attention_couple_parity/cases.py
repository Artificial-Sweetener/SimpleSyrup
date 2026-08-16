# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own the immutable prompt-only standard-UNet parity case."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.visual_prompt_defaults import (
    BASE_NEGATIVE_G,
    BASE_NEGATIVE_L,
    BASE_POSITIVE_G,
    BASE_POSITIVE_L,
    LEFT_BASE_G,
    LEFT_BASE_L,
    REGIONAL_SCENE_G,
    RIGHT_BASE_G,
    RIGHT_BASE_L,
)


@dataclass(frozen=True, slots=True)
class SdxlAttentionCoupleParityCase:
    """Retain generic native-SDXL prompts for one controlled comparison."""

    base_positive_g: str
    base_positive_l: str
    base_negative_g: str
    base_negative_l: str
    left_positive_g: str
    left_positive_l: str
    right_positive_g: str
    right_positive_l: str
    left_negative_g: str
    left_negative_l: str
    right_negative_g: str
    right_negative_l: str


def parity_case() -> SdxlAttentionCoupleParityCase:
    """Return the characterized model-neutral prompt-only two-subject case."""

    return SdxlAttentionCoupleParityCase(
        base_positive_g=(
            f"{BASE_POSITIVE_G}, left role: {LEFT_BASE_G}, right role: {RIGHT_BASE_G}"
        ),
        base_positive_l=BASE_POSITIVE_L,
        base_negative_g=BASE_NEGATIVE_G,
        base_negative_l=BASE_NEGATIVE_L,
        left_positive_g=f"{REGIONAL_SCENE_G}, {LEFT_BASE_G}",
        left_positive_l=LEFT_BASE_L,
        right_positive_g=f"{REGIONAL_SCENE_G}, {RIGHT_BASE_G}",
        right_positive_l=RIGHT_BASE_L,
        left_negative_g=BASE_NEGATIVE_G,
        left_negative_l=BASE_NEGATIVE_L,
        right_negative_g=BASE_NEGATIVE_G,
        right_negative_l=BASE_NEGATIVE_L,
    )


def load_parity_case(path: Path) -> SdxlAttentionCoupleParityCase:
    """Load one complete model-neutral parity prompt case from JSON."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Parity prompt case does not exist.")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Parity prompt case must be one JSON object.")
    values = cast(JsonObject, payload)
    fields = tuple(SdxlAttentionCoupleParityCase.__dataclass_fields__)
    if set(values) != set(fields):
        raise ValueError("Parity prompt case fields must match the case contract.")
    if any(not isinstance(values[field], str) or not values[field] for field in fields):
        raise ValueError("Parity prompt case values must be non-empty strings.")
    return SdxlAttentionCoupleParityCase(
        **{field: cast(str, values[field]) for field in fields}
    )
