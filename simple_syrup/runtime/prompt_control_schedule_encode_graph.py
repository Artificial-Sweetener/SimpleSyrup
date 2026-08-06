# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Orchestrate Prompt-Control model scheduling and SEP prompt encoding."""

from __future__ import annotations

from typing import Any

from ..domain.prompt_batch_parser import DEFAULT_PROMPT_BATCH_SEPARATOR
from ..domain.prompt_control_prompt import PreparedPromptSide, apply_encode_style
from ..services.prompt_control_segment_planning_service import (
    PromptControlSegmentPlan,
    PromptControlSegmentPlanningService,
)
from .prompt_control_graph_adapter import (
    PromptControlGraphAdapter,
    RegionalSegmentEncoding,
)

PROMPT_CONTROL_MISSING_MESSAGE = (
    "Schedule & Encode Prompts requires comfyui-prompt-control. "
    "Install Prompt Control or remove this node from the workflow."
)


class PromptControlScheduleEncodeGraphBuilder:
    """Build scheduled model and hook-aware conditioning graph outputs."""

    planning_service_class = PromptControlSegmentPlanningService
    graph_adapter_class = PromptControlGraphAdapter

    def build(
        self,
        model: Any,
        clip: Any,
        positive_prompt: str,
        negative_prompt: str,
        encode_style: str = "",
    ) -> Any:
        """Return model plus single or SEP-batched conditioning outputs."""

        plan = self.planning_service_class().prepare(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            separator=DEFAULT_PROMPT_BATCH_SEPARATOR,
        )
        adapter = self.graph_adapter_class.load(PROMPT_CONTROL_MISSING_MESSAGE)
        expand: dict[str, dict[str, Any]] = {}
        scheduled_model, encoding_clip = self._sampling_inputs(
            model=model,
            clip=clip,
            plan=plan,
            adapter=adapter,
            expand=expand,
        )
        segment_encodings = self._segment_encodings(
            plan=plan,
            clip=encoding_clip,
            adapter=adapter,
            expand=expand,
        )
        positive = self._encode_side(
            plan.positive,
            segment_encodings=segment_encodings,
            encode_style=encode_style,
            adapter=adapter,
            expand=expand,
            label="positive",
        )
        negative = self._encode_side(
            plan.negative,
            segment_encodings=segment_encodings,
            encode_style=encode_style,
            adapter=adapter,
            expand=expand,
            label="negative",
        )
        return adapter.io.NodeOutput(
            scheduled_model,
            positive,
            negative,
            expand=expand,
        )

    def _sampling_inputs(
        self,
        *,
        model: Any,
        clip: Any,
        plan: PromptControlSegmentPlan,
        adapter: PromptControlGraphAdapter,
        expand: dict[str, dict[str, Any]],
    ) -> tuple[Any, Any]:
        """Keep single prompts global and batched prompts segment-local."""

        if plan.is_batched:
            return model, clip
        return adapter.schedule_global_loras(
            model=model,
            clip=clip,
            positive_tags=plan.positive.chunks[0].lora_tags,
            negative_tags=plan.negative.chunks[0].lora_tags,
            expand=expand,
        )

    def _segment_encodings(
        self,
        *,
        plan: PromptControlSegmentPlan,
        clip: Any,
        adapter: PromptControlGraphAdapter,
        expand: dict[str, dict[str, Any]],
    ) -> tuple[RegionalSegmentEncoding, ...]:
        """Create one shared encoding context per batched segment index."""

        if not plan.is_batched:
            return (RegionalSegmentEncoding(clip=clip),)
        return tuple(
            adapter.clip_with_hooks(
                clip=clip,
                lora_tags=hook.lora_tags,
                expand=expand,
                label=f"segment {index}",
            )
            for index, hook in enumerate(plan.hooks)
        )

    def _encode_side(
        self,
        side: PreparedPromptSide,
        *,
        segment_encodings: tuple[RegionalSegmentEncoding, ...],
        encode_style: str,
        adapter: PromptControlGraphAdapter,
        expand: dict[str, dict[str, Any]],
        label: str,
    ) -> Any:
        """Encode one side with the shared hook plan at each existing index."""

        outputs = [
            adapter.encode_segment(
                segment=segment_encodings[index],
                text=apply_encode_style(encode_style, chunk.text),
                expand=expand,
                label=f"{label} segment {index}",
            )
            for index, chunk in enumerate(side.chunks)
        ]
        for index in range(1, len(outputs)):
            segment = segment_encodings[index]
            if segment.hooks is None:
                continue
            global_companion = adapter.encode_segment(
                segment=segment,
                text=apply_encode_style(encode_style, side.chunks[0].text),
                expand=expand,
                label=f"{label} segment {index} global companion",
            )
            outputs[index] = adapter.attach_global_companion(
                conditioning=outputs[index],
                global_conditioning=global_companion,
                expand=expand,
                label=f"{label} segment {index}",
            )
        return adapter.pack_conditionings(
            outputs,
            expand=expand,
            label=label,
            always_batch=False,
        )
