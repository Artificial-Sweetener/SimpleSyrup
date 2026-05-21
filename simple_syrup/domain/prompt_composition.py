# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prompt text composition helpers."""

from __future__ import annotations


def prefix_prompt(prefix: str, prompt: str) -> str:
    """Return prompt text with a blank-safe comma-separated prefix."""

    prefix_text = prefix.strip()
    prompt_text = prompt.strip()
    if not prefix_text:
        return prompt_text
    if not prompt_text:
        return prefix_text
    return f"{prefix_text}, {prompt_text}"
