# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shape retained attention alpha into controllable feathered mattes."""

from __future__ import annotations

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
        filled = _fill_enclosed_holes(support).float()
        solid = feather_mask(filled, edge_feather)
        return torch.lerp(raw, solid, float(solidity)).clamp(0.0, 1.0)


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
