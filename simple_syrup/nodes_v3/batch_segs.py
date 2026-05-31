# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node wrapper for Batch SEGS."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..domain.segs import batch_segs, to_impact_compatible_segs

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class BatchSEGSV3(_ComfyNodeBase):
    """Expose expandable SEGS batching through Comfy's v3 API."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the Batch SEGS v3 schema."""

        autogrow_template = _comfy_io.Autogrow.TemplatePrefix(
            _comfy_io.SEGS.Input(
                "segs",
                tooltip="SEGS payload to append to the output batch.",
            ),
            prefix="segs",
            min=2,
            max=50,
        )
        return _comfy_io.Schema(
            node_id="SimpleSyrup.BatchSEGS",
            display_name="Batch SEGS",
            category="SimpleSyrup/Detection",
            description="Combines multiple SEGS inputs into one ordered SEGS payload.",
            search_aliases=["batch", "merge", "join", "combine", "segs"],
            inputs=[
                _comfy_io.Autogrow.Input(
                    "segs_inputs",
                    template=autogrow_template,
                    tooltip="Expandable SEGS inputs joined in socket order.",
                ),
            ],
            outputs=[
                _comfy_io.SEGS.Output(
                    "segs",
                    tooltip="Combined SEGS with all input segments in order.",
                ),
            ],
        )

    @classmethod
    def execute(cls, segs_inputs: Mapping[str, object]) -> tuple[object]:
        """Batch provided SEGS inputs in Autogrow order."""

        native = batch_segs(segs_inputs.values())
        return (to_impact_compatible_segs(native),)
