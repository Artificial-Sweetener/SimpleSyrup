# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for batching SEGS."""

from __future__ import annotations

from typing import Any

from ..domain.segs import batch_segs, to_impact_compatible_segs
from ..nodes import tooltips


class BatchSEGS:
    """Combine two SEGS payloads into one ordered SEGS payload."""

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    OUTPUT_TOOLTIPS = (tooltips.BATCH_SEGS_OUTPUT,)
    FUNCTION = "batch"
    CATEGORY = "SimpleSyrup/Detection"
    DESCRIPTION = (
        "Combines two SEGS inputs into one ordered SEGS payload. Chain this node "
        "to batch more than two SEGS sources."
    )
    SEARCH_ALIASES = ["batch", "merge", "join", "combine", "segs"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare legacy ComfyUI inputs for SEGS batching."""

        return {
            "required": {
                "first": ("SEGS", {"tooltip": tooltips.BATCH_SEGS_FIRST}),
                "second": ("SEGS", {"tooltip": tooltips.BATCH_SEGS_SECOND}),
            },
        }

    def batch(self, first: object, second: object) -> tuple[object]:
        """Batch two SEGS payloads in input order."""

        native = batch_segs((first, second))
        return (to_impact_compatible_segs(native),)
