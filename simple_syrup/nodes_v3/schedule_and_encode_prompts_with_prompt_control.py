# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node for Prompt-Control prompt scheduling and encoding."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..runtime.prompt_control_schedule_encode_graph import (
    PromptControlScheduleEncodeGraphBuilder,
)

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

        @classmethod
        def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
            """Return v1-compatible input metadata."""

            raise NotImplementedError

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io
MixedConditioningIO: Any = (
    None if TYPE_CHECKING else _comfy_io.Custom("CONDITIONING,CONDITIONING_BATCH")
)


class ScheduleAndEncodePromptsWithPromptControl(_ComfyNodeBase):
    """Schedule Prompt-Control LoRAs and encode prompts with optional batches."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the Prompt-Control schedule and encode schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl",
            display_name="Schedule & Encode Prompts",
            enable_expand=True,
            category="SimpleSyrup/Conditioning",
            description=(
                "Schedules Prompt-Control LoRAs and encodes prompts, using [SEP] "
                "to create SimpleSyrup conditioning batches."
            ),
            inputs=[
                _comfy_io.Model.Input(
                    "model",
                    raw_link=True,
                    tooltip=(
                        "Model that receives LoRA changes found in Prompt-Control "
                        "prompt tags."
                    ),
                ),
                _comfy_io.Clip.Input(
                    "clip",
                    raw_link=True,
                    tooltip=(
                        "CLIP connection used for scheduled hooks and cleaned "
                        "prompt encoding."
                    ),
                ),
                _comfy_io.String.Input(
                    "encode_style",
                    default="",
                    force_input=True,
                    optional=True,
                    tooltip=(
                        "Encode style text from Prompt Encode Style or "
                        "Prompt Encode Style & Normalization."
                    ),
                ),
                _comfy_io.String.Input(
                    "positive_prompt",
                    multiline=False,
                    default="",
                    tooltip=(
                        "Positive Prompt-Control text. [SEP] creates a "
                        "conditioning batch for SimpleSyrup batch-aware nodes."
                    ),
                ),
                _comfy_io.String.Input(
                    "negative_prompt",
                    multiline=False,
                    default="",
                    tooltip=(
                        "Negative Prompt-Control text. [SEP] creates a "
                        "conditioning batch for SimpleSyrup batch-aware nodes."
                    ),
                ),
            ],
            outputs=[
                _comfy_io.Model.Output(
                    "model",
                    tooltip=(
                        "Model after LoRA tags from positive and negative prompts "
                        "are scheduled."
                    ),
                ),
                MixedConditioningIO.Output(
                    "positive",
                    tooltip="Positive conditioning or SimpleSyrup conditioning batch.",
                ),
                MixedConditioningIO.Output(
                    "negative",
                    tooltip="Negative conditioning or SimpleSyrup conditioning batch.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        clip: Any,
        positive_prompt: str,
        negative_prompt: str,
        encode_style: str = "",
    ) -> Any:
        """Build lazy Prompt-Control graph expansion for prompts."""

        return PromptControlScheduleEncodeGraphBuilder().build(
            model=model,
            clip=clip,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            encode_style=encode_style,
        )
