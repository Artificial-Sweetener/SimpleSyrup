# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shape retained attention alpha into controllable feathered mattes."""

from __future__ import annotations

from importlib import import_module

import torch

from ..masking.mask_components import connected_mask_components
from ..masking.segs_mask_ops import feather_mask


class AttentionMatteService:
    """Blend raw attention alpha toward a solid, hole-filled matte."""

    def shape(
        self,
        *,
        alpha: torch.Tensor,
        support: torch.Tensor,
        solidity: float,
        edge_feather: int,
    ) -> torch.Tensor:
        """Return one finite HW matte with solidity confined to accepted support."""

        raw = alpha.float().clamp(0.0, 1.0)
        if solidity <= 0.0:
            return raw
        topology_radius = _topology_radius(support, edge_feather)
        cohesive = _close_narrow_channels(support, topology_radius)
        filled = _fill_enclosed_holes(cohesive).float()
        solid = feather_mask(filled, edge_feather)
        return torch.lerp(raw, solid, float(solidity)).clamp(0.0, 1.0)


def _topology_radius(support: torch.Tensor, edge_feather: int) -> int:
    """Scale narrow-channel cleanup conservatively with output resolution."""

    height, width = int(support.shape[0]), int(support.shape[1])
    resolution_radius = round(min(height, width) * 0.03)
    return max(1, min(32, max(edge_feather, resolution_radius)))


def _close_narrow_channels(support: torch.Tensor, radius: int) -> torch.Tensor:
    """Close thin exterior branches without filling their broad source opening."""

    height, width = int(support.shape[0]), int(support.shape[1])
    diameter = radius * 2 + 1
    if min(height, width) <= diameter:
        return support.detach().to(dtype=torch.bool)
    active = support.detach().to(device="cpu", dtype=torch.uint8).numpy()
    cv2 = import_module("cv2")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    closed = cv2.morphologyEx(
        active,
        cv2.MORPH_CLOSE,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    cohesive = (closed > 0) | (active > 0)
    return torch.from_numpy(cohesive).to(device=support.device)


def _fill_enclosed_holes(support: torch.Tensor) -> torch.Tensor:
    """Fill inverse components that do not touch the image boundary."""

    active = support.detach().to(device="cpu", dtype=torch.bool)
    inverse = ~active
    height, width = int(active.shape[0]), int(active.shape[1])
    filled = active
    for component in connected_mask_components(inverse):
        box = component.bbox
        touches_edge = (
            box.left == 0 or box.top == 0 or box.right == width or box.bottom == height
        )
        if not touches_edge:
            filled |= component.mask
    return filled.to(device=support.device)


ATTENTION_MATTE_SERVICE = AttentionMatteService()
