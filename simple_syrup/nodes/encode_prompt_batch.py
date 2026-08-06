# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for standard prompt batch encoding."""

from __future__ import annotations

from typing import Any, ClassVar

from ..domain.prompt_batch_parser import DEFAULT_PROMPT_BATCH_SEPARATOR
from ..runtime.conditioning_encoding import ComfyConditioningEncoder
from ..services.prompt_batch_encoding_service import PromptBatchEncodingService


class EncodePromptBatch:
    """Encode separator-delimited prompts into conditioning batches."""

    RETURN_TYPES = ("CONDITIONING_BATCH", "CONDITIONING_BATCH")
    RETURN_NAMES = ("positive", "negative")
    OUTPUT_TOOLTIPS = (
        "Ordered positive conditioning entries for batch-aware consumers.",
        "Ordered negative conditioning entries for batch-aware consumers.",
    )
    FUNCTION = "encode"
    CATEGORY = "SimpleSyrup/Conditioning"
    DESCRIPTION = (
        "Encodes prompts separated by [SEP] or [SEP|name] into matched "
        "conditioning batches, reusing each side's global prompt when a "
        "regional entry is missing."
    )
    SEARCH_ALIASES = ["conditioning batch", "prompt batch", "segs prompts"]

    encoder_class: ClassVar[type[ComfyConditioningEncoder]] = ComfyConditioningEncoder
    service_class: ClassVar[type[PromptBatchEncodingService]] = (
        PromptBatchEncodingService
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare prompt batch encoder inputs."""

        return {
            "required": {
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "Text encoder used to turn each prompt entry into "
                            "conditioning."
                        )
                    },
                ),
                "positive_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Ordered positive prompt entries separated by [SEP] "
                            "or [SEP|name]; the global entry fills missing "
                            "positive regions."
                        ),
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Ordered negative prompt entries separated by [SEP] "
                            "or [SEP|name]; the global entry fills missing "
                            "negative regions."
                        ),
                    },
                ),
                "separator": (
                    "STRING",
                    {
                        "default": DEFAULT_PROMPT_BATCH_SEPARATOR,
                        "tooltip": (
                            "Text marker that separates prompt entries. With the "
                            "default [SEP], use [SEP|name] to add an organizational "
                            "label."
                        ),
                    },
                ),
            }
        }

    def encode(
        self,
        clip: Any,
        positive_prompt: str,
        negative_prompt: str,
        separator: str,
    ) -> tuple[object, object]:
        """Encode positive and negative prompt batches."""

        return self.service_class(self.encoder_class()).encode(
            clip=clip,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            separator=separator,
        )
