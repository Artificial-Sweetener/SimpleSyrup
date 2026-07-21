# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Orchestrate Prompt Control SEP conditioning-batch graph expansion."""

from __future__ import annotations

from typing import Any

from ..domain.prompt_control_prompt import PreparedPromptSide
from ..services.prompt_control_segment_planning_service import (
    PromptControlSegmentPlanningService,
)
from .prompt_control_graph_adapter import PromptControlGraphAdapter

PROMPT_CONTROL_MISSING_MESSAGE = (
    "Encode Prompt Batch w/ Prompt Control requires comfyui-prompt-control. "
    "Install Prompt Control or use Encode Prompt Batch."
)


class PromptControlBatchGraphBuilder:
    """Build segment-local hook-aware Prompt Control conditioning batches."""

    planning_service_class = PromptControlSegmentPlanningService
    graph_adapter_class = PromptControlGraphAdapter

    def build(
        self,
        clip: Any,
        positive_prompt: str,
        negative_prompt: str,
        separator: str,
    ) -> Any:
        """Return positive and negative conditioning-batch graph links."""

        plan = self.planning_service_class().prepare(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            separator=separator,
        )
        adapter = self.graph_adapter_class.load(PROMPT_CONTROL_MISSING_MESSAGE)
        expand: dict[str, dict[str, Any]] = {}
        segment_clips = tuple(
            adapter.clip_with_hooks(
                clip=clip,
                lora_tags=hook.lora_tags,
                expand=expand,
                label=f"segment {index}",
            )
            for index, hook in enumerate(plan.hooks)
        )
        positive = self._encode_side(
            plan.positive,
            segment_clips=segment_clips,
            adapter=adapter,
            expand=expand,
            label="positive",
        )
        negative = self._encode_side(
            plan.negative,
            segment_clips=segment_clips,
            adapter=adapter,
            expand=expand,
            label="negative",
        )
        return adapter.io.NodeOutput(positive, negative, expand=expand)

    def _encode_side(
        self,
        side: PreparedPromptSide,
        *,
        segment_clips: tuple[Any, ...],
        adapter: PromptControlGraphAdapter,
        expand: dict[str, dict[str, Any]],
        label: str,
    ) -> Any:
        """Encode and pack all existing chunks from one prompt side."""

        outputs = [
            adapter.encode_segment(
                clip=segment_clips[index],
                text=chunk.text,
                expand=expand,
                label=f"{label} segment {index}",
            )
            for index, chunk in enumerate(side.chunks)
        ]
        return adapter.pack_conditionings(
            outputs,
            expand=expand,
            label=label,
            always_batch=True,
        )
