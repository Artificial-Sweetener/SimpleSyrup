# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plan aligned Prompt-Control SEP segments without runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.prompt_control_prompt import (
    PreparedPromptChunk,
    PreparedPromptSide,
    PromptSegmentHookPlan,
    prepare_prompt_side,
)
from ..domain.prompt_segment_alignment import (
    PromptSideAlignment,
    build_prompt_segment_alignment,
)


@dataclass(frozen=True)
class PromptControlSegmentPlan:
    """Store two prompt sides and the LoRA hooks shared at each position."""

    positive: PreparedPromptSide
    negative: PreparedPromptSide
    hooks: tuple[PromptSegmentHookPlan, ...]

    @property
    def is_batched(self) -> bool:
        """Return whether either prompt side contains multiple SEP segments."""

        return len(self.positive.chunks) > 1 or len(self.negative.chunks) > 1


class PromptControlSegmentPlanningService:
    """Prepare prompt sides and align their segment-local LoRA schedules."""

    def prepare(
        self,
        *,
        positive_prompt: str,
        negative_prompt: str,
        separator: str,
    ) -> PromptControlSegmentPlan:
        """Return matched prompt sides and one shared hook plan per position."""

        authored_positive = prepare_prompt_side(positive_prompt, separator)
        authored_negative = prepare_prompt_side(negative_prompt, separator)
        alignment = build_prompt_segment_alignment(
            positive_count=len(authored_positive.chunks),
            negative_count=len(authored_negative.chunks),
        )
        positive = self._align_side(authored_positive, alignment.positive)
        negative = self._align_side(authored_negative, alignment.negative)
        hooks = tuple(
            PromptSegmentHookPlan(
                lora_tags=self._combined_lora_tags(positive, negative, index)
            )
            for index in range(alignment.segment_count)
        )
        return PromptControlSegmentPlan(
            positive=positive,
            negative=negative,
            hooks=hooks,
        )

    def _align_side(
        self,
        side: PreparedPromptSide,
        alignment: PromptSideAlignment,
    ) -> PreparedPromptSide:
        """Build global-context regions without repeating global LoRA tags."""

        chunks = tuple(
            self._effective_chunk(
                authored=source.authored,
                source=source.resolve(side.chunks),
            )
            for source in alignment.sources
        )
        return PreparedPromptSide(chunks=chunks)

    def _effective_chunk(
        self,
        *,
        authored: bool,
        source: PreparedPromptChunk,
    ) -> PreparedPromptChunk:
        """Strip tags only when a chunk was synthesized from global text."""

        if authored:
            return source
        return PreparedPromptChunk(text=source.text, lora_tags="")

    def _combined_lora_tags(
        self,
        positive: PreparedPromptSide,
        negative: PreparedPromptSide,
        index: int,
    ) -> str:
        """Join existing positive then negative tags for one segment index."""

        tags: list[str] = []
        if positive.chunks[index].lora_tags:
            tags.append(positive.chunks[index].lora_tags)
        if negative.chunks[index].lora_tags:
            tags.append(negative.chunks[index].lora_tags)
        return "\n".join(tags)
