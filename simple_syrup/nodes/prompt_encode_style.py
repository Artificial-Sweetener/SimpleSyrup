# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for Prompt Control encode-style tags."""

from __future__ import annotations

from typing import Any

from ..domain.prompt_style import ENCODE_STYLE_VALUES, format_style_tag


class PromptEncodeStyle:
    """Build Prompt Control STYLE tags from encode-style selections."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("style_tag",)
    OUTPUT_TOOLTIPS = ("Prompt Control STYLE tag text for prompt encoding workflows.",)
    FUNCTION = "build"
    CATEGORY = "SimpleSyrup/Prompt"
    DESCRIPTION = "Builds a Prompt Control STYLE tag from an encode-style selection."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare Prompt Control encode-style selection inputs."""

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
            }
        }

    def build(self, encode_style: str) -> tuple[str]:
        """Return a Prompt Control STYLE tag."""

        return (format_style_tag(encode_style),)
