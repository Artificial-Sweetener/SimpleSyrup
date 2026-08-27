# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Extract, score, and retain attention-region components."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..domain.attention_region_capture import AttentionRegionControls
from ..domain.attention_region_evidence import AttentionConceptEvidence
from ..domain.segs import BoundingBox
from ..masking.mask_components import connected_mask_components
from .attention_region_instance_splitting import (
    ATTENTION_INSTANCE_SPLITTING_SERVICE,
)


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
        for cohesive_region in connected_mask_components(evidence.support):
            partitions = ATTENTION_INSTANCE_SPLITTING_SERVICE.partition(
                alpha=evidence.alpha,
                support=cohesive_region.mask,
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
        if controls.keep_only == 0 or len(candidates) <= controls.keep_only:
            return tuple(candidates)
        ranked = sorted(
            candidates, key=lambda value: _rank_key(value, controls.keep_by)
        )
        return tuple(ranked[: controls.keep_only])


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
