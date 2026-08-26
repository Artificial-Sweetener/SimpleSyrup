# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Model ordered sampling stages along one spatial provenance path."""

from __future__ import annotations

from dataclasses import dataclass

from .attention_spatial_transform import AttentionSpatialTransform
from .graph_provenance import GraphLink


@dataclass(frozen=True, slots=True)
class AttentionSamplerStage:
    """Describe one sampler's patch and conditioning authority."""

    sampler_node_id: str
    model_owner_node_id: str
    model_link: GraphLink
    positive_link: GraphLink
    upstream_link: GraphLink
    upstream_kind: str
    forward_transforms: tuple[AttentionSpatialTransform, ...] = ()
    source_aspect: float | None = None
    capture_supported: bool = True
    unsupported_reason: str | None = None

    def __post_init__(self) -> None:
        """Require unsupported stages to explain why capture cannot be projected."""

        if self.capture_supported and self.unsupported_reason is not None:
            raise ValueError("Supported attention sampler stages cannot have a reason.")
        if not self.capture_supported and not self.unsupported_reason:
            raise ValueError("Unsupported attention sampler stages require a reason.")
        if self.source_aspect is not None and self.source_aspect <= 0.0:
            raise ValueError("Attention sampler source aspect must be positive.")


@dataclass(frozen=True, slots=True)
class AttentionSamplerSelection:
    """Bind a selected stage to its one-based chronological position."""

    stage: AttentionSamplerStage
    stage_number: int
    stage_count: int
    was_clamped: bool


@dataclass(frozen=True, slots=True)
class AttentionSamplerLineage:
    """Hold sampling stages ordered from oldest to direct provenance."""

    stages: tuple[AttentionSamplerStage, ...]

    def __post_init__(self) -> None:
        """Require at least one uniquely identified stage."""

        identities = tuple(stage.sampler_node_id for stage in self.stages)
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("Attention sampler lineage must contain unique stages.")

    def select(self, requested_stage: int) -> AttentionSamplerSelection:
        """Resolve one-based selection with 0/-1 aliases for direct provenance."""

        if type(requested_stage) is not int or requested_stage < -1:
            raise ValueError("Attention sampler stage must be -1 or greater.")
        count = len(self.stages)
        if requested_stage in (-1, 0):
            return AttentionSamplerSelection(self.stages[-1], count, count, False)
        selected_number = min(requested_stage, count)
        return AttentionSamplerSelection(
            self.stages[selected_number - 1],
            selected_number,
            count,
            requested_stage > count,
        )
