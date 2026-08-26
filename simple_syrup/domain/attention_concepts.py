# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Parse explicit attention concepts without interpreting prompt language."""

from __future__ import annotations


def parse_attention_concepts(value: str) -> tuple[str, ...]:
    """Return canonical concepts separated only by vertical bars."""

    if not isinstance(value, str):
        raise TypeError("Attention concepts must be text.")
    return tuple(part.strip() for part in value.split("|") if part.strip())
