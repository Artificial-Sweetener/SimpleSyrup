# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Domain model for per-segment conditioning batches."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypeAlias

Conditioning: TypeAlias = Any


@dataclass(frozen=True)
class ConditioningBatch:
    """Store ordered conditioning entries for per-SEG selection."""

    entries: tuple[Conditioning, ...]

    def __post_init__(self) -> None:
        """Reject batches that cannot select a conditioning."""

        if not self.entries:
            raise ValueError(
                "conditioning batch must contain at least one conditioning."
            )

    def select(self, index: int) -> Conditioning:
        """Return the entry for an index, reusing the last entry as fallback."""

        if index < 0:
            raise ValueError("conditioning batch index must be non-negative.")
        return self.entries[min(index, len(self.entries) - 1)]

    def append(self, conditioning: Conditioning) -> ConditioningBatch:
        """Return a new batch with one conditioning appended."""

        return ConditioningBatch((*self.entries, conditioning))


def split_prompt_batch(text: str, separator: str = "[SEP]") -> tuple[str, ...]:
    """Split prompt text into ordered chunks using a configurable separator."""

    if separator == "":
        raise ValueError("separator must not be empty.")
    pattern = rf"\s*{re.escape(separator)}\s*"
    return tuple(re.split(pattern, text))


def select_conditioning(
    conditioning: Conditioning | ConditioningBatch,
    index: int,
) -> Conditioning:
    """Select per-index conditioning or broadcast normal conditioning unchanged."""

    if isinstance(conditioning, ConditioningBatch):
        return conditioning.select(index)
    if index < 0:
        raise ValueError("conditioning batch index must be non-negative.")
    return conditioning
