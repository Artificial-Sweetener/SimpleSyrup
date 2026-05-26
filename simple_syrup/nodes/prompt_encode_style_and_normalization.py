# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for Prompt Control encode-style normalization tags."""

from __future__ import annotations

from typing import Any

from ..domain.prompt_style import (
    ENCODE_STYLE_VALUES,
    NORMALIZATION_VALUES,
    format_style_tag_with_normalization,
)


class PromptEncodeStyleAndNormalization:
    """Build STYLE tags from encode-style and normalization selections."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("encode_style",)
    OUTPUT_TOOLTIPS = (
        "Prompt Control encode style text with the selected normalization behavior.",
    )
    FUNCTION = "build"
    CATEGORY = "SimpleSyrup/Prompt"
    DESCRIPTION = (
        "Builds a Prompt Control STYLE tag from encode-style and normalization "
        "selections."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare Prompt Control encode-style and normalization selection inputs."""

        return {
            "required": {
                "encode_style": (
                    list(ENCODE_STYLE_VALUES),
                    {
                        "default": "A1111",
                        "tooltip": (
                            "Prompt Control encoding style to write into the STYLE tag."
                        ),
                    },
                ),
                "normalization": (
                    list(NORMALIZATION_VALUES),
                    {
                        "default": "none",
                        "tooltip": (
                            "Prompt weight normalization mode written into the "
                            "STYLE tag."
                        ),
                    },
                ),
            }
        }

    def build(self, encode_style: str, normalization: str) -> tuple[str]:
        """Return a Prompt Control STYLE tag."""

        return (format_style_tag_with_normalization(encode_style, normalization),)
