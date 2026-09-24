# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define renderer-neutral SEG preview documents."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .segs import CropRegion


@dataclass(frozen=True)
class AtlasPlacement:
    """Locate one region mask inside the packed mask atlas."""

    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class SegPreviewRegion:
    """Describe one interactive region and its packed mask geometry."""

    region_id: str
    index: int
    label: str
    confidence: float
    active_area: int
    color: str
    crop: CropRegion
    atlas: AtlasPlacement


@dataclass(frozen=True)
class SegPreviewDocument:
    """Carry bounded image assets and interaction metadata to a UI adapter."""

    source_width: int
    source_height: int
    preview_width: int
    preview_height: int
    image: torch.Tensor
    atlas: torch.Tensor
    region_images: tuple[torch.Tensor, ...]
    regions: tuple[SegPreviewRegion, ...]
