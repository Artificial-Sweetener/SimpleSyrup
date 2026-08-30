# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node for exposing every mapped prompt attention region."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..domain.attention_region_capture import AttentionEvidenceMode
from ..services.attention_region_node_service import ATTENTION_REGION_NODE_SERVICE
from .attention_region_inputs import (
    attention_region_control_inputs,
    attention_region_controls,
)

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        hidden: ClassVar[Any]
        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class AllPromptAttentionSEGSV3(_ComfyNodeBase):
    """Expose separate overlapping regions for all readable positive concepts."""

    GRAPH_PASSTHROUGH_OUTPUTS = {0: "image"}

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the all-prompt downstream attention-SEGS contract."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.AllPromptAttentionSEGS",
            display_name="All Prompt Attention SEGS",
            category="SimpleSyrup/Detection",
            description=(
                "Returns separate overlapping soft SEGS for every mapped concept "
                "in the positive prompt that produced the image."
            ),
            search_aliases=[
                "all attention heatmaps",
                "prompt insight",
                "token regions",
            ],
            hidden=[_comfy_io.Hidden.unique_id],
            inputs=[
                _comfy_io.Image.Input(
                    "image",
                    tooltip=(
                        "Image whose graph provenance identifies the upstream sampler; "
                        "the image is returned unchanged."
                    ),
                ),
                *attention_region_control_inputs(
                    _comfy_io,
                    evidence_mode_default=AttentionEvidenceMode.RAW,
                ),
            ],
            outputs=[
                _comfy_io.Image.Output("image", tooltip="Unchanged connected image."),
                _comfy_io.SEGS.Output(
                    "segs",
                    tooltip="Separate labeled overlapping SEGS for prompt concepts.",
                    is_output_list=True,
                ),
                _comfy_io.Mask.Output(
                    "mask",
                    tooltip="Soft union of every retained prompt attention region.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: object,
        sampler_stage: int,
        capture_start: float,
        capture_end: float,
        minimum_strength: float,
        minimum_consensus: float,
        geometry_recall: float,
        split_sensitivity: float,
        instance_recall: float,
        minimum_region_size: int,
        keep_only: int,
        keep_by: str,
        combine_segs: bool,
        matte_solidity: float,
        edge_feather: int,
        capture_profile: str,
        evidence_mode: str,
    ) -> tuple[object, object, object]:
        """Consume all prompt maps and render separate overlapping SEGS."""

        del sampler_stage
        result = ATTENTION_REGION_NODE_SERVICE.for_image(
            request_node_id=str(cls.hidden.unique_id),
            image=image,
            controls=attention_region_controls(
                capture_start=capture_start,
                capture_end=capture_end,
                minimum_strength=minimum_strength,
                minimum_consensus=minimum_consensus,
                geometry_recall=geometry_recall,
                split_sensitivity=split_sensitivity,
                instance_recall=instance_recall,
                minimum_region_size=minimum_region_size,
                keep_only=keep_only,
                keep_by=keep_by,
                combine_segs=combine_segs,
                matte_solidity=matte_solidity,
                edge_feather=edge_feather,
                capture_profile=capture_profile,
                evidence_mode=evidence_mode,
            ),
        )
        return result.image, result.segs, result.mask
