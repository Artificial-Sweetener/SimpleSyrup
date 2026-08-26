# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose explicitly requested upstream attention concepts as SEGS."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..domain.attention_concepts import parse_attention_concepts
from ..services.attention_region_node_service import ATTENTION_REGION_NODE_SERVICE
from .attention_region_inputs import (
    attention_region_control_inputs,
    attention_region_controls,
)

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        hidden: ClassVar[Any]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class ConceptAttentionSEGSV3(_ComfyNodeBase):
    """Return attention-derived instances for explicitly named concepts."""

    GRAPH_PASSTHROUGH_OUTPUTS = {0: "image"}

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the downstream concept-attention SEGS contract."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.ConceptAttentionSEGS",
            display_name="Concept Attention SEGS",
            category="SimpleSyrup/Detection",
            description=(
                "Returns regions associated with explicit concepts from a selected "
                "sampler in the connected image's provenance."
            ),
            search_aliases=[
                "attention heatmap",
                "prompt segmentation",
                "attention mask",
            ],
            hidden=[_comfy_io.Hidden.unique_id],
            inputs=[
                _comfy_io.Image.Input(
                    "image",
                    tooltip=(
                        "Image whose graph provenance identifies the sampling chain; "
                        "the image is returned unchanged."
                    ),
                ),
                _comfy_io.String.Input(
                    "concepts",
                    multiline=True,
                    default="subject",
                    tooltip=(
                        "Enter one or more concepts separated by |, such as "
                        "girl | pink hair | cat."
                    ),
                ),
                *attention_region_control_inputs(_comfy_io),
            ],
            outputs=[
                _comfy_io.Image.Output("image", tooltip="Unchanged connected image."),
                _comfy_io.SEGS.Output(
                    "segs",
                    tooltip="Labeled attention-derived instances for the concepts.",
                    is_output_list=True,
                ),
                _comfy_io.Mask.Output(
                    "mask", tooltip="Union of all retained concept instances."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: object,
        concepts: str,
        sampler_stage: int,
        capture_start: float,
        capture_end: float,
        minimum_strength: float,
        minimum_consensus: float,
        split_sensitivity: float,
        minimum_region_size: int,
        keep_only: int,
        keep_by: str,
        combine_segs: bool,
        matte_solidity: float,
        edge_feather: int,
        capture_profile: str,
    ) -> tuple[object, object, object]:
        """Consume shared capture evidence and render requested concept SEGS."""

        del sampler_stage
        if not parse_attention_concepts(concepts):
            raise ValueError("Concept Attention SEGS requires at least one concept.")
        result = ATTENTION_REGION_NODE_SERVICE.for_image(
            request_node_id=str(cls.hidden.unique_id),
            image=image,
            controls=attention_region_controls(
                capture_start=capture_start,
                capture_end=capture_end,
                minimum_strength=minimum_strength,
                minimum_consensus=minimum_consensus,
                split_sensitivity=split_sensitivity,
                minimum_region_size=minimum_region_size,
                keep_only=keep_only,
                keep_by=keep_by,
                combine_segs=combine_segs,
                matte_solidity=matte_solidity,
                edge_feather=edge_feather,
                capture_profile=capture_profile,
            ),
        )
        return result.image, result.segs, result.mask
