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
        "Model with single-prompt LoRAs applied; SEP-local LoRAs stay on conditioning.",
        "Positive conditioning or SimpleSyrup conditioning batch.",
        "Negative conditioning or SimpleSyrup conditioning batch.",
    )
    FUNCTION = "execute"
    CATEGORY = "SimpleSyrup/Conditioning"
    DESCRIPTION = (
        "Schedules Prompt-Control LoRAs and encodes prompts. [SEP] creates "
        "matched conditioning batches using global text for missing regions."
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
                            "Positive Prompt-Control text; [SEP] creates ordered "
                            "entries, and global text fills missing positive regions."
                        ),
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Negative Prompt-Control text; [SEP] creates ordered "
                            "entries, and global text fills missing negative regions."
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
