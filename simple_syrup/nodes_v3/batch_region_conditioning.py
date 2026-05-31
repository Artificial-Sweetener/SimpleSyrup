# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node wrapper for batching regional conditioning."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..domain.conditioning_batch import batch_conditioning

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io
ConditioningBatchIO: Any = (
    None if TYPE_CHECKING else _comfy_io.Custom("CONDITIONING_BATCH")
)


class BatchRegionConditioningV3(_ComfyNodeBase):
    """Expose expandable mixed conditioning batching through Comfy's v3 API."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the Batch Region Conditioning v3 schema."""

        conditioning_input = _comfy_io.MultiType.Input(
            "conditioning",
            [_comfy_io.Conditioning, ConditioningBatchIO],
            tooltip="Conditioning or conditioning batch to append in socket order.",
        )
        autogrow_template = _comfy_io.Autogrow.TemplatePrefix(
            conditioning_input,
            prefix="conditioning",
            min=2,
            max=50,
        )
        return _comfy_io.Schema(
            node_id="SimpleSyrup.BatchRegionConditioning",
            display_name="Batch Region Conditioning",
            category="SimpleSyrup/Conditioning",
            description=(
                "Combines CONDITIONING and CONDITIONING_BATCH inputs into one "
                "ordered regional conditioning batch."
            ),
            search_aliases=[
                "batch",
                "conditioning batch",
                "region conditioning",
                "segs prompts",
            ],
            inputs=[
                _comfy_io.Autogrow.Input(
                    "conditioning_inputs",
                    template=autogrow_template,
                    tooltip=(
                        "Expandable conditioning inputs flattened in socket order."
                    ),
                ),
            ],
            outputs=[
                ConditioningBatchIO.Output(
                    "batch",
                    tooltip=(
                        "Conditioning batch containing all input entries in order."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, conditioning_inputs: Mapping[str, object]) -> tuple[object]:
        """Batch conditioning inputs in Autogrow order."""

        return (batch_conditioning(tuple(conditioning_inputs.values())),)
