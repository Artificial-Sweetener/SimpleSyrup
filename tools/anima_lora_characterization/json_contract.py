# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Narrow dynamic JSON values at P0.7 artifact boundaries."""

from __future__ import annotations

import math
from typing import cast

from tools.comfy_api import JsonObject


def object_value(value: object, label: str) -> JsonObject:
    """Narrow one JSON object with string keys."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object.")
    return cast(JsonObject, value)


def array_value(value: object, label: str) -> list[object]:
    """Narrow one JSON array."""

    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array.")
    return value


def string_list(value: object, label: str) -> list[str]:
    """Narrow one JSON string array."""

    items = array_value(value, label)
    if not all(isinstance(item, str) for item in items):
        raise ValueError(f"{label} must contain strings.")
    return [item for item in items if isinstance(item, str)]


def text_value(value: object, label: str, *, empty: bool = False) -> str:
    """Narrow one required JSON string."""

    if not isinstance(value, str) or (not empty and not value):
        raise ValueError(f"{label} must be a string.")
    return value


def boolean_value(value: object, label: str) -> bool:
    """Narrow one JSON boolean."""

    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean.")
    return value


def integer_value(value: object, label: str) -> int:
    """Narrow one nonnegative JSON integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer.")
    return value


def number_value(value: object, label: str) -> float:
    """Narrow one finite nonnegative JSON number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative.")
    return result
