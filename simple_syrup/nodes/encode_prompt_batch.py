# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for standard prompt batch encoding."""

from __future__ import annotations

from typing import Any, ClassVar

from ..domain.conditioning_batch import split_prompt_batch
from ..runtime.conditioning_encoding import ComfyConditioningEncoder


class EncodePromptBatch:
    """Encode separator-delimited prompts into conditioning batches."""

    RETURN_TYPES = ("CONDITIONING_BATCH", "CONDITIONING_BATCH")
    RETURN_NAMES = ("positive", "negative")
    OUTPUT_TOOLTIPS = (
        "Positive conditioning entries selected by SEGS order.",
        "Negative conditioning entries selected by SEGS order.",
    )
    FUNCTION = "encode"
    CATEGORY = "SimpleSyrup/Conditioning"
    DESCRIPTION = (
        "Encodes [SEP]-separated prompts into per-segment conditioning batches."
    )
    SEARCH_ALIASES = ["conditioning batch", "prompt batch", "segs prompts"]

    encoder_class: ClassVar[type[ComfyConditioningEncoder]] = ComfyConditioningEncoder

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
                            "Positive prompts in SEGS order, separated by [SEP]."
                        ),
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Negative prompts in SEGS order, separated by [SEP]."
                        ),
                    },
                ),
                "separator": (
                    "STRING",
                    {
                        "default": "[SEP]",
                        "tooltip": "Text marker that separates prompt entries.",
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

        encoder = self.encoder_class()
        positive_chunks = split_prompt_batch(positive_prompt, separator)
        negative_chunks = split_prompt_batch(negative_prompt, separator)
        return (
            encoder.encode_batch(clip, positive_chunks),
            encoder.encode_batch(clip, negative_chunks),
        )
