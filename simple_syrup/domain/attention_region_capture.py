# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define immutable requests and plans for attention-region capture."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .attention_spatial_transform import AttentionSpatialTransform
from .graph_provenance import GraphLink


class AttentionRegionRequestKind(StrEnum):
    """Identify one public attention-region operation."""

    CONCEPT_SEGS = "concept_segs"
    ALL_PROMPT_SEGS = "all_prompt_segs"
    REGION_MASK = "region_mask"
    MASKED_CONDITIONING = "masked_conditioning"


class AttentionCaptureProfile(StrEnum):
    """Select the density of attention observations retained during sampling."""

    FAST = "fast"
    BALANCED = "balanced"
    EXHAUSTIVE = "exhaustive"


class AttentionEvidenceMode(StrEnum):
    """Select honest inspection or derived concept-isolation evidence."""

    CONCEPT = "concept isolation"
    RAW = "raw attention"


@dataclass(frozen=True, slots=True)
class AttentionRegionControls:
    """Hold validated attention-native capture and region-shaping controls."""

    capture_start: float
    capture_end: float
    minimum_strength: float
    minimum_consensus: float
    split_sensitivity: float
    minimum_region_size: int
    profile: AttentionCaptureProfile
    instance_recall: float = 0.65
    geometry_recall: float = 0.85
    keep_only: int = 0
    keep_by: str = "largest size"
    combine_segs: bool = False
    matte_solidity: float = 0.0
    edge_feather: int = 8
    evidence_mode: AttentionEvidenceMode = AttentionEvidenceMode.CONCEPT

    def __post_init__(self) -> None:
        """Require normalized ranges and a non-empty capture interval."""

        normalized = (
            self.capture_start,
            self.capture_end,
            self.minimum_strength,
            self.minimum_consensus,
            self.split_sensitivity,
            self.instance_recall,
            self.geometry_recall,
            self.matte_solidity,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in normalized
        ):
            raise TypeError("Attention-region controls must be real numbers.")
        if not 0.0 <= self.capture_start < self.capture_end <= 1.0:
            raise ValueError("Attention capture start must be below end within 0..1.")
        if not 0.0 <= self.minimum_strength <= 1.0:
            raise ValueError("Minimum attention strength must be within 0..1.")
        if not 0.0 <= self.minimum_consensus <= 1.0:
            raise ValueError("Minimum attention consensus must be within 0..1.")
        if not 0.0 <= self.split_sensitivity <= 1.0:
            raise ValueError("Attention split sensitivity must be within 0..1.")
        if not 0.0 <= self.instance_recall <= 1.0:
            raise ValueError("Attention instance recall must be within 0..1.")
        if not 0.0 <= self.geometry_recall <= 1.0:
            raise ValueError("Attention geometry recall must be within 0..1.")
        if type(self.minimum_region_size) is not int or self.minimum_region_size < 1:
            raise ValueError("Minimum attention region size must be positive.")
        if type(self.keep_only) is not int or self.keep_only < 0:
            raise ValueError("Attention keep_only must be non-negative.")
        if self.keep_by not in ("largest size", "highest confidence"):
            raise ValueError("Attention keep_by has an invalid policy.")
        if type(self.combine_segs) is not bool:
            raise TypeError("Attention combine_segs must be boolean.")
        if not 0.0 <= self.matte_solidity <= 1.0:
            raise ValueError("Attention matte solidity must be within 0..1.")
        if type(self.edge_feather) is not int or self.edge_feather < 0:
            raise ValueError("Attention edge feather must be non-negative.")
        if not isinstance(self.profile, AttentionCaptureProfile):
            raise TypeError("Attention capture profile has an invalid type.")
        if not isinstance(self.evidence_mode, AttentionEvidenceMode):
            raise TypeError("Attention evidence mode has an invalid type.")


@dataclass(frozen=True, slots=True)
class AttentionRegionRequest:
    """Bind one public node request to its queries and capture controls."""

    node_id: str
    kind: AttentionRegionRequestKind
    queries: tuple[str, ...]
    controls: AttentionRegionControls
    sampler_stage: int = 1
    spatial_transforms: tuple[AttentionSpatialTransform, ...] = ()

    def __post_init__(self) -> None:
        """Require stable node identity and canonical non-empty query strings."""

        if not self.node_id.strip():
            raise ValueError("Attention-region request node id cannot be empty.")
        if not isinstance(self.kind, AttentionRegionRequestKind):
            raise TypeError("Attention-region request kind has an invalid type.")
        if any(not query or query != query.strip() for query in self.queries):
            raise ValueError("Attention-region queries must be canonical strings.")
        if self.kind is AttentionRegionRequestKind.ALL_PROMPT_SEGS:
            if self.queries:
                raise ValueError(
                    "All-prompt attention requests cannot contain queries."
                )
        elif not self.queries:
            raise ValueError("Concept and mask attention requests require concepts.")
        if type(self.sampler_stage) is not int or self.sampler_stage < -1:
            raise ValueError("Attention sampler stage must be -1 or greater.")
        if any(
            not isinstance(transform, AttentionSpatialTransform)
            for transform in self.spatial_transforms
        ):
            raise TypeError("Attention request spatial transforms are invalid.")


@dataclass(frozen=True, slots=True)
class AttentionCapturePlan:
    """Describe one coalesced sampler capture and its graph rewrite authority."""

    sampler_node_id: str
    model_owner_node_id: str
    model_input_name: str
    model_link: GraphLink
    positive_link: GraphLink
    requests: tuple[AttentionRegionRequest, ...]
    prompt_text: str | None = None
    clip_link: GraphLink | None = None
    source_aspect: float | None = None

    def __post_init__(self) -> None:
        """Require canonical unique requests and complete graph-edge identity."""

        if not self.sampler_node_id or not self.model_owner_node_id:
            raise ValueError("Attention capture plan node ids cannot be empty.")
        if not self.model_input_name:
            raise ValueError("Attention capture plan model input cannot be empty.")
        if self.source_aspect is not None and self.source_aspect <= 0.0:
            raise ValueError("Attention capture source aspect must be positive.")
        request_ids = tuple(request.node_id for request in self.requests)
        if not request_ids or request_ids != tuple(sorted(set(request_ids))):
            raise ValueError("Attention capture requests must be unique and ordered.")

    @property
    def capture_node_id(self) -> str:
        """Return a collision-resistant deterministic injected node id."""

        return f"__simple_syrup_attention_capture__{self.sampler_node_id}"
