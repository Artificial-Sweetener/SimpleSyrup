# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plan aligned Prompt-Control SEP segments without runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.prompt_control_prompt import (
    PreparedPromptSide,
    PromptSegmentHookPlan,
    prepare_prompt_side,
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
        """Return a deterministic plan without padding either prompt side."""

        positive = prepare_prompt_side(positive_prompt, separator)
        negative = prepare_prompt_side(negative_prompt, separator)
        hook_count = max(len(positive.chunks), len(negative.chunks))
        hooks = tuple(
            PromptSegmentHookPlan(
                lora_tags=self._combined_lora_tags(positive, negative, index)
            )
            for index in range(hook_count)
        )
        return PromptControlSegmentPlan(
            positive=positive,
            negative=negative,
            hooks=hooks,
        )

    def _combined_lora_tags(
        self,
        positive: PreparedPromptSide,
        negative: PreparedPromptSide,
        index: int,
    ) -> str:
        """Join existing positive then negative tags for one segment index."""

        tags: list[str] = []
        if index < len(positive.chunks) and positive.chunks[index].lora_tags:
            tags.append(positive.chunks[index].lora_tags)
        if index < len(negative.chunks) and negative.chunks[index].lora_tags:
            tags.append(negative.chunks[index].lora_tags)
        return "\n".join(tags)
