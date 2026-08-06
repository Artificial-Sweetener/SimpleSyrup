# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Parse authored prompt text into ordered batch entries."""

from __future__ import annotations

import re

DEFAULT_PROMPT_BATCH_SEPARATOR = "[SEP]"
_NAMED_DEFAULT_SEPARATOR_PATTERN = r"\[SEP(?:\|[^\r\n\]]+)?\]"


def split_prompt_batch(
    text: str,
    separator: str = DEFAULT_PROMPT_BATCH_SEPARATOR,
) -> tuple[str, ...]:
    """Split prompt text while discarding default-separator labels."""

    if separator == "":
        raise ValueError("separator must not be empty.")
    pattern = rf"\s*(?:{_separator_pattern(separator)})\s*"
    return tuple(re.split(pattern, text))


def _separator_pattern(separator: str) -> str:
    """Return labeled default grammar or an escaped custom separator pattern."""

    if separator == DEFAULT_PROMPT_BATCH_SEPARATOR:
        return _NAMED_DEFAULT_SEPARATOR_PATTERN
    return re.escape(separator)
