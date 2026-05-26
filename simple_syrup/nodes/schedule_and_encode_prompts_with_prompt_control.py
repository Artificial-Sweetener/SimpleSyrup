# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Legacy ComfyUI node for Prompt-Control prompt scheduling and encoding."""

from __future__ import annotations

from typing import Any

from ..runtime.prompt_control_schedule_encode_graph import (
    PromptControlScheduleEncodeGraphBuilder,
)


class ScheduleAndEncodePromptsWithPromptControl:
    """Schedule Prompt-Control LoRAs and encode prompts with optional batches."""

    RETURN_TYPES = (
        "MODEL",
        "CONDITIONING,CONDITIONING_BATCH",
        "CONDITIONING,CONDITIONING_BATCH",
    )
    RETURN_NAMES = ("model", "positive", "negative")
    OUTPUT_TOOLTIPS = (
        "Model after LoRA tags from positive and negative prompts are scheduled.",
        "Positive conditioning or SimpleSyrup conditioning batch.",
        "Negative conditioning or SimpleSyrup conditioning batch.",
    )
    FUNCTION = "execute"
    CATEGORY = "SimpleSyrup/Conditioning"
    DESCRIPTION = (
        "Schedules Prompt-Control LoRAs and encodes prompts, using [SEP] to "
        "create SimpleSyrup conditioning batches."
    )
    SEARCH_ALIASES = ["prompt control", "schedule prompts", "encode prompts"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare Prompt-Control schedule and encode inputs."""

        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "rawLink": True,
                        "tooltip": (
                            "Model that receives LoRA changes found in "
                            "Prompt-Control prompt tags."
                        ),
                    },
                ),
                "clip": (
                    "CLIP",
                    {
                        "rawLink": True,
                        "tooltip": (
                            "CLIP connection used for scheduled hooks and "
                            "cleaned prompt encoding."
                        ),
                    },
                ),
                "positive_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Positive Prompt-Control text. [SEP] creates a "
                            "conditioning batch for SimpleSyrup batch-aware nodes."
                        ),
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Negative Prompt-Control text. [SEP] creates a "
                            "conditioning batch for SimpleSyrup batch-aware nodes."
                        ),
                    },
                ),
            },
            "optional": {
                "encode_style": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": (
                            "Encode style text from Prompt Encode Style or "
                            "Prompt Encode Style & Normalization."
                        ),
                    },
                ),
            },
        }

    def execute(
        self,
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
