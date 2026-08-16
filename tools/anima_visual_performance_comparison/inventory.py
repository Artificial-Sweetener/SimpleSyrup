# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load external model selections and prompts for an Anima visual comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True, slots=True)
class AnimaVisualInventory:
    """Retain externally selected model roles and authored prompt segments."""

    diffusion_model: str
    text_encoder: str
    vae: str
    regional_lora: str
    global_positive: str
    left_positive: str
    right_positive: str
    global_negative: str
    left_negative: str
    right_negative: str

    @classmethod
    def load(cls, path: Path) -> AnimaVisualInventory:
        """Decode one complete external JSON object."""

        value: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("Anima visual inventory must be a JSON object.")
        payload = cast(dict[str, object], value)
        fields = tuple(cls.__dataclass_fields__)
        if set(payload) != set(fields):
            raise ValueError("Anima visual inventory fields must match its contract.")
        return cls(**{field: _text(payload[field]) for field in fields})


def _text(value: object) -> str:
    """Narrow one external inventory value to non-empty text."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Anima visual inventory values must be non-empty text.")
    return value
