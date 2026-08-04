# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Publish interactive SEG preview assets through ComfyUI temporary storage."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol

import torch

from ..services.simple_preview_segs_service import SegPreviewDocument

SEG_PREVIEW_UI_KEY = "simple_syrup_segs_preview"


@dataclass(frozen=True)
class SegPreviewPublication:
    """Carry native gallery images beside overlay interaction metadata."""

    manifest: dict[str, object]
    images: tuple[dict[str, str], ...]


class SegPreviewAssetPublisher(Protocol):
    """Publish one backend preview document as a JSON-compatible UI payload."""

    def publish(self, document: SegPreviewDocument) -> SegPreviewPublication:
        """Return native gallery references and overlay interaction metadata."""


class ComfySegPreviewAssetPublisher:
    """Store preview tensors using ComfyUI's authoritative image helper."""

    def publish(self, document: SegPreviewDocument) -> SegPreviewPublication:
        """Publish native RGBA regions plus the overlay image and mask atlas."""

        comfy_api: Any = import_module("comfy_api.latest")
        image_ref = _publish_one(comfy_api, document.image)
        atlas_ref = _publish_one(comfy_api, document.atlas)
        region_refs = tuple(
            _publish_one(comfy_api, image) for image in document.region_images
        )
        manifest: dict[str, object] = {
            "version": 1,
            "source": {
                "width": document.source_width,
                "height": document.source_height,
            },
            "preview": {
                "width": document.preview_width,
                "height": document.preview_height,
                "image": image_ref,
            },
            "atlas": {
                "width": int(document.atlas.shape[2]),
                "height": int(document.atlas.shape[1]),
                "image": atlas_ref,
            },
            "regions": [
                {
                    "id": region.region_id,
                    "index": region.index,
                    "label": region.label,
                    "confidence": region.confidence,
                    "area": region.active_area,
                    "color": region.color,
                    "crop": {
                        "x": region.crop.left,
                        "y": region.crop.top,
                        "width": region.crop.width,
                        "height": region.crop.height,
                    },
                    "atlas": {
                        "x": region.atlas.left,
                        "y": region.atlas.top,
                        "width": region.atlas.width,
                        "height": region.atlas.height,
                    },
                }
                for region in document.regions
            ],
        }
        return SegPreviewPublication(manifest=manifest, images=region_refs)


def _publish_one(comfy_api: Any, image: torch.Tensor) -> dict[str, str]:
    """Publish one IMAGE tensor and validate ComfyUI's returned reference."""

    payload: object = comfy_api.UI.PreviewImage(image).as_dict()
    if not isinstance(payload, dict):
        raise TypeError("ComfyUI returned an invalid SEG preview asset payload.")
    images = payload.get("images")
    if not isinstance(images, (list, tuple)) or len(images) != 1:
        raise TypeError("ComfyUI must publish exactly one SEG preview asset.")
    reference = images[0]
    if not isinstance(reference, dict):
        raise TypeError("ComfyUI returned an invalid SEG preview image reference.")
    required = ("filename", "subfolder", "type")
    if any(not isinstance(reference.get(name), str) for name in required):
        raise TypeError("ComfyUI SEG preview image reference is incomplete.")
    return {name: str(reference[name]) for name in required}
