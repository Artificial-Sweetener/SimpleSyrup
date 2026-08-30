# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Extract, score, and retain attention-region components."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch

from ..domain.attention_region_capture import (
    AttentionEvidenceMode,
    AttentionRegionControls,
)
from ..domain.attention_region_evidence import AttentionConceptEvidence
from ..domain.segs import BoundingBox
from ..masking.mask_components import connected_mask_components
from .attention_region_component_cohesion import (
    ATTENTION_COMPONENT_COHESION_SERVICE,
)
from .attention_region_instance_splitting import (
    ATTENTION_INSTANCE_SPLITTING_SERVICE,
)

logger = logging.getLogger(__name__)

MINIMUM_COMPACT_DOMINANT_AREA_SHARE = 0.65
MAXIMUM_COMPACT_RUNNER_UP_AREA_SHARE = 0.5
MINIMUM_COMPACT_BOUNDING_BOX_FILL = 0.35
MAXIMUM_COMPACT_COMPONENT_ELONGATION = 1.75


@dataclass(frozen=True, slots=True)
class AttentionRegionComponent:
    """Represent one retained concept instance before matte shaping."""

    label: str
    alpha: torch.Tensor
    support: torch.Tensor
    bbox: BoundingBox
    area: int
    confidence: float


class AttentionComponentService:
    """Own per-concept component filtering and deterministic retention."""

    def retained(
        self,
        evidence: AttentionConceptEvidence,
        controls: AttentionRegionControls,
    ) -> tuple[AttentionRegionComponent, ...]:
        """Return valid components under the requested per-concept policy."""

        candidates: list[AttentionRegionComponent] = []
        eligible_supports = tuple(
            region.mask.to(device=evidence.support.device)
            for region in connected_mask_components(evidence.support)
            if int(region.mask.count_nonzero().item()) >= controls.minimum_region_size
        )
        cohesive_supports = ATTENTION_COMPONENT_COHESION_SERVICE.group(
            eligible_supports
        )
        for cohesive_support in cohesive_supports:
            partitions = ATTENTION_INSTANCE_SPLITTING_SERVICE.partition(
                alpha=evidence.alpha,
                support=cohesive_support,
                minimum_strength=controls.minimum_strength,
                sensitivity=controls.split_sensitivity,
            )
            for partition in partitions:
                area = int(partition.count_nonzero().item())
                if area < controls.minimum_region_size:
                    continue
                confidence = float(evidence.confidence[partition].mean().item())
                candidates.append(
                    AttentionRegionComponent(
                        evidence.label,
                        evidence.alpha * partition.to(dtype=evidence.alpha.dtype),
                        partition,
                        _active_bounds(partition),
                        area,
                        confidence,
                    )
                )
        retained = tuple(candidates)
        if controls.evidence_mode is AttentionEvidenceMode.CONCEPT:
            diagnostic_components = sorted(
                retained,
                key=lambda component: -component.area,
            )[:10]
            logger.debug(
                "Evaluated concept attention components: label=%s count=%d "
                "largest_areas=%s largest_scores=%s",
                evidence.label,
                len(retained),
                tuple(component.area for component in diagnostic_components),
                tuple(
                    round(_component_evidence_score(component), 4)
                    for component in diagnostic_components
                ),
            )
            retained = _retain_reliable_concept_components(
                retained,
                controls.instance_recall,
            )
            retained = _retain_compact_dominant_component(
                retained,
                controls.instance_recall,
            )
        if controls.keep_only == 0 or len(retained) <= controls.keep_only:
            return retained
        ranked = sorted(retained, key=lambda value: _rank_key(value, controls.keep_by))
        return tuple(ranked[: controls.keep_only])


def _retain_reliable_concept_components(
    components: tuple[AttentionRegionComponent, ...],
    instance_recall: float,
) -> tuple[AttentionRegionComponent, ...]:
    """Reject weak disconnected context relative to the best supported region."""

    if len(components) < 2:
        return components
    scores = tuple(_component_evidence_score(component) for component in components)
    strongest = max(scores)
    if strongest <= 0.0:
        return components
    floor = strongest * (1.0 - instance_recall)
    return tuple(
        component
        for component, score in zip(components, scores, strict=True)
        if score >= floor
    )


def _retain_compact_dominant_component(
    components: tuple[AttentionRegionComponent, ...],
    instance_recall: float,
) -> tuple[AttentionRegionComponent, ...]:
    """Reject disconnected pockmarks around one compact dominant concept body."""

    if len(components) < 2 or instance_recall >= 1.0:
        return components
    ranked = sorted(
        components,
        key=lambda component: (
            -component.area,
            component.bbox.top,
            component.bbox.left,
        ),
    )
    dominant = ranked[0]
    runner_up = ranked[1]
    total_area = sum(component.area for component in ranked)
    bounding_box_area = dominant.bbox.width * dominant.bbox.height
    shorter_side = min(dominant.bbox.width, dominant.bbox.height)
    longer_side = max(dominant.bbox.width, dominant.bbox.height)
    if (
        dominant.area / total_area < MINIMUM_COMPACT_DOMINANT_AREA_SHARE
        or runner_up.area / dominant.area > MAXIMUM_COMPACT_RUNNER_UP_AREA_SHARE
        or dominant.area / bounding_box_area < MINIMUM_COMPACT_BOUNDING_BOX_FILL
        or longer_side / shorter_side > MAXIMUM_COMPACT_COMPONENT_ELONGATION
    ):
        return components
    logger.debug(
        "Selected compact dominant attention component: label=%s retained_area=%d "
        "discarded_components=%d",
        dominant.label,
        dominant.area,
        len(components) - 1,
    )
    return (dominant,)


def _component_evidence_score(component: AttentionRegionComponent) -> float:
    """Score evidence quality independently of component area."""

    values = component.alpha[component.support].float()
    peak = values.amax()
    strongest_count = max(1, math.ceil(int(values.numel()) * 0.25))
    strongest_mean = values.topk(strongest_count).values.mean()
    return float((peak * 0.65 + strongest_mean * 0.35).item())


def _rank_key(
    component: AttentionRegionComponent, keep_by: str
) -> tuple[float, int, int]:
    """Return deterministic descending evidence or area ordering."""

    primary = (
        -component.confidence if keep_by == "highest confidence" else -component.area
    )
    return primary, component.bbox.top, component.bbox.left


def _active_bounds(active: torch.Tensor) -> BoundingBox:
    """Return the minimal full-image bounds around one active partition."""

    coordinates = active.nonzero(as_tuple=False)
    if int(coordinates.shape[0]) < 1:
        raise ValueError("Attention instance partition requires an active pixel.")
    top = int(coordinates[:, 0].min().item())
    bottom = int(coordinates[:, 0].max().item()) + 1
    left = int(coordinates[:, 1].min().item())
    right = int(coordinates[:, 1].max().item()) + 1
    return BoundingBox(left, top, right, bottom)


ATTENTION_COMPONENT_SERVICE = AttentionComponentService()
