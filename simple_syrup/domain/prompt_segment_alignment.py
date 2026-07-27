# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plan authored and global-fallback positions across two prompt sides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

SegmentValue = TypeVar("SegmentValue")


@dataclass(frozen=True)
class PromptSegmentSource:
    """Identify one effective segment's authored source position."""

    source_index: int
    authored: bool

    def resolve(self, values: tuple[SegmentValue, ...]) -> SegmentValue:
        """Return the authored value selected for this effective position."""

        return values[self.source_index]


@dataclass(frozen=True)
class PromptSideAlignment:
    """Describe effective positions for one authored prompt side."""

    authored_count: int
    sources: tuple[PromptSegmentSource, ...]

    def materialize(
        self,
        values: tuple[SegmentValue, ...],
    ) -> tuple[SegmentValue, ...]:
        """Resolve effective values while rejecting a mismatched authored side."""

        if len(values) != self.authored_count:
            raise ValueError(
                "prompt alignment expected "
                f"{self.authored_count} authored segments but received {len(values)}."
            )
        return tuple(source.resolve(values) for source in self.sources)


@dataclass(frozen=True)
class PromptSegmentAlignment:
    """Store matched effective positions for positive and negative prompts."""

    positive: PromptSideAlignment
    negative: PromptSideAlignment

    @property
    def segment_count(self) -> int:
        """Return the shared number of effective prompt positions."""

        return len(self.positive.sources)


def build_prompt_segment_alignment(
    *,
    positive_count: int,
    negative_count: int,
) -> PromptSegmentAlignment:
    """Align prompt sides by filling missing positions from each global entry."""

    if positive_count < 1 or negative_count < 1:
        raise ValueError("each prompt side must contain at least one authored segment.")
    segment_count = max(positive_count, negative_count)
    return PromptSegmentAlignment(
        positive=_build_side_alignment(positive_count, segment_count),
        negative=_build_side_alignment(negative_count, segment_count),
    )


def _build_side_alignment(
    authored_count: int,
    segment_count: int,
) -> PromptSideAlignment:
    """Return authored positions followed by global-entry fallback positions."""

    return PromptSideAlignment(
        authored_count=authored_count,
        sources=tuple(
            PromptSegmentSource(
                source_index=index if index < authored_count else 0,
                authored=index < authored_count,
            )
            for index in range(segment_count)
        ),
    )
