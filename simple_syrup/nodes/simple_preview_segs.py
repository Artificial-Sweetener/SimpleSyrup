# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for the interactive Simple SEGS inspector."""

from __future__ import annotations

from typing import Any, ClassVar

from ..domain.segs import coerce_segs
from ..runtime.seg_preview_assets import (
    SEG_PREVIEW_UI_KEY,
    ComfySegPreviewAssetPublisher,
)
from ..services.simple_preview_segs_service import SimplePreviewSEGSService


class SimplePreviewSEGS:
    """Expose an interactive overlay and grid inspector for one IMAGE and SEGS."""

    service_class: ClassVar[type[SimplePreviewSEGSService]] = SimplePreviewSEGSService
    publisher_class: ClassVar[type[ComfySegPreviewAssetPublisher]] = (
        ComfySegPreviewAssetPublisher
    )

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    OUTPUT_TOOLTIPS = ("The original SEGS passed through without modification.",)
    FUNCTION = "preview"
    CATEGORY = "SimpleSyrup/Preview"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Interactively inspects SEGS over the original image or in a selectable grid."
    )
    SEARCH_ALIASES = ["segs", "regions", "inspect", "overlay", "preview"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare the source image and matching SEGS inputs."""

        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Original image described by the connected SEGS."},
                ),
                "segs": (
                    "SEGS",
                    {"tooltip": "Regions to inspect over the original image."},
                ),
            }
        }

    def preview(self, image: object, segs: object) -> dict[str, object]:
        """Publish an interactive preview while returning the input SEGS unchanged."""

        native_segs = coerce_segs(segs)
        document = self.service_class().build(image=image, segs=native_segs)
        manifest = self.publisher_class().publish(document)
        return {
            "ui": {SEG_PREVIEW_UI_KEY: [manifest]},
            "result": (segs,),
        }
