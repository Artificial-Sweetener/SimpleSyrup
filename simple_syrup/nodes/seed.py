# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for reusable seed values."""

from __future__ import annotations


class Seed:
    """Expose ComfyUI's native seed widget as a reusable integer output."""

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    OUTPUT_TOOLTIPS = ("Same seed value for wiring into multiple nodes.",)
    FUNCTION = "execute"
    CATEGORY = "SimpleSyrup/Utilities"
    DESCRIPTION = "Provides a reusable seed value with ComfyUI seed controls."
    SEARCH_ALIASES = ["seed", "random seed"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        """Declare the native ComfyUI seed input contract."""

        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": (
                            "Seed value to reuse across nodes. Matching seed and "
                            "settings make random choices repeatable."
                        ),
                    },
                )
            }
        }

    def execute(self, seed: int) -> tuple[int]:
        """Return the selected seed unchanged."""

        return (seed,)
