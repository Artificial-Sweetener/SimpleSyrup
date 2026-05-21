# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI nodes for constructing SimpleSyrup conditioning batches."""

from __future__ import annotations

from typing import Any

from ..domain.conditioning_batch import ConditioningBatch


class ConditioningBatchStart:
    """Start a conditioning batch from one normal conditioning value."""

    RETURN_TYPES = ("CONDITIONING_BATCH",)
    RETURN_NAMES = ("batch",)
    OUTPUT_TOOLTIPS = (
        "Conditioning batch with one entry, ready to align with the first SEGS item.",
    )
    FUNCTION = "pack"
    CATEGORY = "SimpleSyrup/Conditioning"
    DESCRIPTION = "Starts a per-segment conditioning batch."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare the first conditioning input."""

        return {
            "required": {
                "conditioning": (
                    "CONDITIONING",
                    {
                        "tooltip": (
                            "Conditioning for the first SEGS item in a per-region "
                            "batch."
                        )
                    },
                ),
            }
        }

    def pack(self, conditioning: Any) -> tuple[ConditioningBatch]:
        """Return a new conditioning batch with one entry."""

        return (ConditioningBatch((conditioning,)),)


class ConditioningBatchAppend:
    """Append one normal conditioning value to an existing batch."""

    RETURN_TYPES = ("CONDITIONING_BATCH",)
    RETURN_NAMES = ("batch",)
    OUTPUT_TOOLTIPS = ("Conditioning batch with the new entry added at the end.",)
    FUNCTION = "append"
    CATEGORY = "SimpleSyrup/Conditioning"
    DESCRIPTION = "Appends a conditioning entry to a per-segment batch."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare the existing batch and conditioning inputs."""

        return {
            "required": {
                "batch": (
                    "CONDITIONING_BATCH",
                    {"tooltip": ("Existing per-region batch to extend in SEGS order.")},
                ),
                "conditioning": (
                    "CONDITIONING",
                    {
                        "tooltip": (
                            "Conditioning to add as the next per-region batch entry."
                        )
                    },
                ),
            }
        }

    def append(
        self,
        batch: ConditioningBatch,
        conditioning: Any,
    ) -> tuple[ConditioningBatch]:
        """Return a new batch with one additional conditioning entry."""

        return (batch.append(conditioning),)
