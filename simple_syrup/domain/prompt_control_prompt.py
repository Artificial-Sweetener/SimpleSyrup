# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare Prompt-Control prompt text for scheduling and encoding."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .conditioning_batch import split_prompt_batch

PROMPT_TEXT_PATTERN = r"(?:^|>)([^<]+)(?=<|$)"
LORA_TAG_PATTERN = r"<[^>]*>"


@dataclass(frozen=True)
class PreparedPromptChunk:
    """Store one prompt chunk's cleaned text and scheduling tags."""

    text: str
    lora_tags: str


@dataclass(frozen=True)
class PreparedPromptSide:
    """Store ordered prompt chunks and all scheduling tags for one prompt side."""

    chunks: tuple[PreparedPromptChunk, ...]
    lora_tags: str


def extract_prompt_text(text: str) -> str:
    """Return prompt text outside angle-bracket Prompt-Control tags."""

    return _extract_all_matches(text, PROMPT_TEXT_PATTERN)


def extract_lora_tags(text: str) -> str:
    """Return angle-bracket Prompt-Control tags joined with newlines."""

    return _extract_all_matches(text, LORA_TAG_PATTERN)


def prepare_prompt_side(text: str, separator: str) -> PreparedPromptSide:
    """Split a prompt side into cleaned chunks and aggregate LoRA tags."""

    chunks = tuple(
        PreparedPromptChunk(
            text=extract_prompt_text(chunk),
            lora_tags=extract_lora_tags(chunk),
        )
        for chunk in split_prompt_batch(text, separator)
    )
    lora_tags = "\n".join(chunk.lora_tags for chunk in chunks if chunk.lora_tags)
    return PreparedPromptSide(chunks=chunks, lora_tags=lora_tags)


def apply_encode_style(encode_style: str, prompt_text: str) -> str:
    """Prepend Prompt-Control encode style text exactly as provided."""

    if not encode_style:
        return prompt_text
    return f"{encode_style}{prompt_text}"


def _extract_all_matches(text: str, pattern: str) -> str:
    """Match Comfy's RegexExtract All Matches behavior."""

    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return ""
    if isinstance(matches[0], tuple):
        return "\n".join(match[0] for match in matches)
    return "\n".join(matches)
