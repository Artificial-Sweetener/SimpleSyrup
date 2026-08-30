# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node for attention-derived regional conditioning of a later sampler."""

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


class AttentionMaskedConditioningV3(_ComfyNodeBase):
    """Mask supplied conditioning with attention captured from an earlier sampler."""

    GRAPH_PASSTHROUGH_OUTPUTS = {0: "latent"}

    @classmethod
    def define_schema(cls) -> Any:
        """Declare later-pass attention-masked conditioning inputs and outputs."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.AttentionMaskedConditioning",
            display_name="Attention Masked Conditioning",
            category="SimpleSyrup/Conditioning",
            description=(
                "Masks supplied conditioning to regions discovered from an earlier "
                "sampler so it can control a later sampling pass."
            ),
            search_aliases=[
                "regional conditioning",
                "attention conditioning",
                "masked prompt",
            ],
            hidden=[_comfy_io.Hidden.unique_id],
            inputs=[
                _comfy_io.Latent.Input(
                    "latent",
                    tooltip=(
                        "Latent produced by the sampler used for localization; it is "
                        "returned unchanged for the later sampler."
                    ),
                ),
                _comfy_io.Conditioning.Input(
                    "conditioning",
                    tooltip="Conditioning to apply only inside the discovered regions.",
                ),
                _comfy_io.String.Input(
                    "concepts",
                    multiline=True,
                    default="subject",
                    tooltip=("Concepts used to discover regions, separated by |."),
                ),
                _comfy_io.Float.Input(
                    "conditioning_strength",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.05,
                    tooltip="Strength of the supplied conditioning inside the mask.",
                ),
                *attention_region_control_inputs(_comfy_io),
            ],
            outputs=[
                _comfy_io.Latent.Output(
                    "latent", tooltip="Unchanged localization latent."
                ),
                _comfy_io.Conditioning.Output(
                    "conditioning",
                    tooltip=(
                        "Supplied conditioning carrying the attention-derived mask."
                    ),
                ),
                _comfy_io.Mask.Output(
                    "mask",
                    tooltip="Soft latent-resolution mask attached to the conditioning.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent: object,
        conditioning: object,
        concepts: str,
        conditioning_strength: float,
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
        """Return later-pass conditioning masked by earlier-sampler attention."""

        del sampler_stage
        if not parse_attention_concepts(concepts):
            raise ValueError("Attention Masked Conditioning requires a concept.")
        result = ATTENTION_REGION_NODE_SERVICE.for_latent(
            request_node_id=str(cls.hidden.unique_id),
            latent=latent,
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
        masked = ATTENTION_REGION_NODE_SERVICE.mask_conditioning(
            conditioning,
            result.mask,
            conditioning_strength,
        )
        return result.latent, masked, result.mask
