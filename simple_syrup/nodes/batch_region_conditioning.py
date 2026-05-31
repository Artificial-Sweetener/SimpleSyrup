# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for batching regional conditioning."""

from __future__ import annotations

from typing import Any

from ..domain.conditioning_batch import batch_conditioning
from ..nodes import tooltips


class BatchRegionConditioning:
    """Combine conditioning and conditioning batches for regional detailing."""

    RETURN_TYPES = ("CONDITIONING_BATCH",)
    RETURN_NAMES = ("batch",)
    OUTPUT_TOOLTIPS = (tooltips.BATCH_REGION_CONDITIONING_OUTPUT,)
    FUNCTION = "batch"
    CATEGORY = "SimpleSyrup/Conditioning"
    DESCRIPTION = (
        "Combines two CONDITIONING or CONDITIONING_BATCH inputs into one ordered "
        "regional conditioning batch. Chain this node to batch more sources."
    )
    SEARCH_ALIASES = [
        "batch",
        "conditioning batch",
        "region conditioning",
        "segs prompts",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare legacy ComfyUI inputs for regional conditioning batching."""

        return {
            "required": {
                "first": (
                    "CONDITIONING,CONDITIONING_BATCH",
                    {"tooltip": tooltips.BATCH_REGION_CONDITIONING_FIRST},
                ),
                "second": (
                    "CONDITIONING,CONDITIONING_BATCH",
                    {"tooltip": tooltips.BATCH_REGION_CONDITIONING_SECOND},
                ),
            },
        }

    def batch(self, first: Any, second: Any) -> tuple[object]:
        """Batch two conditioning inputs in input order."""

        return (batch_conditioning((first, second)),)
