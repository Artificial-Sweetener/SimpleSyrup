# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Orchestrate attention evidence, components, mattes, and SEGS packaging."""

from __future__ import annotations

import torch

from ..domain.attention_region_capture import AttentionRegionControls
from ..domain.attention_region_maps import CapturedAttentionMap
from ..domain.segs import BoundingBox, CropRegion, NativeSegs, Segment
from .attention_region_components import ATTENTION_COMPONENT_SERVICE
from .attention_region_evidence import ATTENTION_EVIDENCE_AGGREGATOR
from .attention_region_matte import ATTENTION_MATTE_SERVICE


class AttentionRegionRenderingService:
    """Render captured maps through independently owned shaping policies."""

    def render(
        self,
        *,
        maps: tuple[CapturedAttentionMap, ...],
        controls: AttentionRegionControls,
        height: int,
        width: int,
        image: torch.Tensor | None = None,
        batch_index: int = 0,
    ) -> tuple[NativeSegs, torch.Tensor]:
        """Return Impact-compatible SEGS and their full-image union mask."""

        _validate_geometry(height, width, image)
        evidence_values = ATTENTION_EVIDENCE_AGGREGATOR.aggregate(
            maps=maps,
            controls=controls,
            height=height,
            width=width,
            batch_index=batch_index,
        )
        segments: list[Segment] = []
        union = torch.zeros((1, height, width), dtype=torch.float32)
        for evidence in evidence_values:
            components = ATTENTION_COMPONENT_SERVICE.retained(evidence, controls)
            shaped = tuple(
                (
                    component,
                    ATTENTION_MATTE_SERVICE.shape(
                        alpha=component.alpha,
                        support=component.support,
                        solidity=controls.matte_solidity,
                        edge_feather=controls.edge_feather,
                    ),
                )
                for component in components
            )
            if controls.combine_segs and shaped:
                combined_mask = torch.stack(tuple(mask for _item, mask in shaped)).amax(
                    dim=0
                )
                confidence = max(item.confidence for item, _mask in shaped)
                segments.append(
                    _segment_from_mask(evidence.label, combined_mask, confidence, image)
                )
                union = torch.maximum(union, combined_mask.unsqueeze(0))
                continue
            for component, matte in shaped:
                segments.append(
                    _segment_from_mask(
                        component.label, matte, component.confidence, image
                    )
                )
                union = torch.maximum(union, matte.unsqueeze(0))
        return ((height, width), tuple(segments)), union


def _segment_from_mask(
    label: str,
    full_mask: torch.Tensor,
    confidence: float,
    image: torch.Tensor | None,
) -> Segment:
    """Crop one shaped full-image matte into a native SEG."""

    bbox = _active_bounds(full_mask > 0.0)
    region = CropRegion(bbox.left, bbox.top, bbox.right, bbox.bottom)
    cropped_mask = full_mask[region.top : region.bottom, region.left : region.right]
    cropped_image: object | None = None
    if image is not None:
        cropped_image = image[
            0, region.top : region.bottom, region.left : region.right, :
        ]
    return Segment(
        cropped_image=cropped_image,
        cropped_mask=cropped_mask,
        confidence=max(0.0, min(1.0, confidence)),
        crop_region=region,
        bbox=bbox,
        label=label,
    )


def _active_bounds(active: torch.Tensor) -> BoundingBox:
    """Return the minimal box containing every nonzero matte pixel."""

    coordinates = active.nonzero(as_tuple=False)
    if int(coordinates.shape[0]) < 1:
        raise ValueError("Attention-region bounds require an active pixel.")
    top = int(coordinates[:, 0].min().item())
    bottom = int(coordinates[:, 0].max().item()) + 1
    left = int(coordinates[:, 1].min().item())
    right = int(coordinates[:, 1].max().item()) + 1
    return BoundingBox(left, top, right, bottom)


def _validate_geometry(
    height: int,
    width: int,
    image: torch.Tensor | None,
) -> None:
    """Require positive output geometry and an optional matching BHWC image."""

    if type(height) is not int or height < 1 or type(width) is not int or width < 1:
        raise ValueError("Attention-region output dimensions must be positive.")
    if image is None:
        return
    if (
        not isinstance(image, torch.Tensor)
        or image.ndim != 4
        or int(image.shape[0]) < 1
        or int(image.shape[1]) != height
        or int(image.shape[2]) != width
    ):
        raise ValueError(
            "Attention-region image must be a matching non-empty BHWC tensor."
        )


ATTENTION_REGION_RENDERING_SERVICE = AttentionRegionRenderingService()
