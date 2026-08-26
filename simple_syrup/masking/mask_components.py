# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Extract deterministic connected components from binary mask regions."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import torch

from ..domain.segs import BoundingBox


@dataclass(frozen=True, slots=True)
class MaskComponent:
    """Represent one connected mask component and its full-image bbox."""

    bbox: BoundingBox
    mask: torch.Tensor


def connected_mask_components(active_mask: torch.Tensor) -> tuple[MaskComponent, ...]:
    """Return spatially ordered 8-connected components from one HW mask."""

    if not isinstance(active_mask, torch.Tensor) or active_mask.ndim != 2:
        raise ValueError("active_mask must be an HW tensor.")
    active = active_mask.detach().to(device="cpu", dtype=torch.uint8).numpy()
    if not active.any():
        return ()
    cv2 = import_module("cv2")
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        active,
        connectivity=8,
    )
    components: list[MaskComponent] = []
    for label in range(1, int(component_count)):
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        components.append(
            MaskComponent(
                bbox=BoundingBox(left, top, left + width, top + height),
                mask=torch.from_numpy(labels == label),
            )
        )
    return tuple(
        sorted(components, key=lambda value: (value.bbox.top, value.bbox.left))
    )
