# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for bounded scale-factor values."""

from __future__ import annotations

from typing import Any

from . import tooltips

SCALE_FACTOR_DEFAULT = 1.5
SCALE_FACTOR_MIN = 1.0
SCALE_FACTOR_MAX = 5.0
SCALE_FACTOR_STEP = 0.1


def scale_factor_options(
    default: float = SCALE_FACTOR_DEFAULT,
    tooltip: str = tooltips.DETAIL_SCALE_FACTOR,
) -> dict[str, object]:
    """Return ComfyUI widget options for scale-factor controls."""

    return {
        "default": default,
        "min": SCALE_FACTOR_MIN,
        "max": SCALE_FACTOR_MAX,
        "step": SCALE_FACTOR_STEP,
        "tooltip": tooltip,
    }


class ScaleFactor:
    """Expose a bounded float value for scale-factor inputs."""

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("scale_factor",)
    OUTPUT_TOOLTIPS = (tooltips.SCALE_FACTOR_OUTPUT,)
    FUNCTION = "get_value"
    CATEGORY = "SimpleSyrup/Primitives"
    DESCRIPTION = "Provides a bounded multiplier for scaling."
    SEARCH_ALIASES = ["scale", "scale factor", "float", "primitive"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare the bounded scale-factor value input."""

        return {
            "required": {
                "value": (
                    "FLOAT",
                    scale_factor_options(tooltip=tooltips.SCALE_FACTOR_VALUE),
                ),
            }
        }

    def get_value(self, value: object) -> tuple[float]:
        """Return the provided scale-factor value."""

        if isinstance(value, (int, float, str)):
            return (float(value),)
        raise TypeError("Scale Factor requires 'value' to be a float.")
