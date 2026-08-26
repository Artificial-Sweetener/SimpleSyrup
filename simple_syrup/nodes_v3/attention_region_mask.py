# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node for querying upstream attention as a latent-resolution mask."""

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
        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class AttentionRegionMaskV3(_ComfyNodeBase):
    """Return a reusable mask derived from an upstream sampler's attention."""

    GRAPH_PASSTHROUGH_OUTPUTS = {0: "latent"}

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the latent attention-region mask contract."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.AttentionRegionMask",
            display_name="Attention Region Mask",
            category="SimpleSyrup/Masking",
            description=(
                "Returns a reusable latent-resolution mask for concepts attended "
                "by the sampler that produced the connected latent."
            ),
            search_aliases=["latent attention mask", "prompt mask", "region mask"],
            hidden=[_comfy_io.Hidden.unique_id],
            inputs=[
                _comfy_io.Latent.Input(
                    "latent",
                    tooltip=(
                        "Latent whose graph provenance identifies the upstream "
                        "sampler; the latent is returned unchanged."
                    ),
                ),
                _comfy_io.String.Input(
                    "concepts",
                    multiline=True,
                    default="subject",
                    tooltip=(
                        "Concepts whose attention becomes the mask, separated by |."
                    ),
                ),
                *attention_region_control_inputs(_comfy_io),
            ],
            outputs=[
                _comfy_io.Latent.Output(
                    "latent", tooltip="Unchanged connected latent."
                ),
                _comfy_io.Mask.Output(
                    "mask",
                    tooltip="Soft latent-resolution union of matched regions.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent: object,
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
    ) -> tuple[object, object]:
        """Consume matching maps and return a reusable latent-resolution mask."""

        del sampler_stage
        if not parse_attention_concepts(concepts):
            raise ValueError("Attention Region Mask requires at least one concept.")
        result = ATTENTION_REGION_NODE_SERVICE.for_latent(
            request_node_id=str(cls.hidden.unique_id),
            latent=latent,
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
        return result.latent, result.mask
