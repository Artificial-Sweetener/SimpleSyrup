# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for Ultralytics SEGS detection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import torch

from ..domain.segs import (
    KEEP_BY_OPTIONS,
    SORT_ORDER_OPTIONS,
    NativeSegs,
)
from ..masking.segs_mask_ops import iter_single_images, validate_image_batch
from ..runtime.ultralytics_loader import UltralyticsDetectorModel
from ..services.segs_detection_service import (
    SegsDetectionService,
)
from ..services.segs_output_service import (
    CombinedSegsResult,
    build_combined_segs_result,
    finalize_detector_segs_output,
)


class DetectSEGSWithUltralytics:
    """Detect regions using Simple Detector SEGS-style controls."""

    RETURN_TYPES = ("SEGS", "MASK")
    RETURN_NAMES = ("segs", "mask")
    OUTPUT_IS_LIST = (True, False)
    OUTPUT_TOOLTIPS = (
        "Detected regions as separate or combined SEGS based on combine_segs.",
        "Combined detected area as a standard ComfyUI mask.",
    )
    FUNCTION = "detect"
    CATEGORY = "SimpleSyrup/Detection"
    DESCRIPTION = (
        "Detects regions with an Ultralytics model and returns individual SEGS, "
        "combined SEGS when requested, and a combined mask."
    )
    SEARCH_ALIASES = ["ultralytics", "yolo", "segs", "detector"]

    service_class: ClassVar[type[SegsDetectionService]] = SegsDetectionService
    combined_builder: ClassVar[
        Callable[[object, NativeSegs, float], CombinedSegsResult]
    ] = build_combined_segs_result

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        """Declare ComfyUI inputs for detector-to-SEGS conversion."""

        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Image to search for detectable regions."},
                ),
                "detector_model": (
                    "DETECTOR_MODEL",
                    {
                        "tooltip": (
                            "Ultralytics model that finds boxes or masks in the "
                            "input image."
                        )
                    },
                ),
                "confidence_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Minimum detection confidence required to keep a region."
                        ),
                    },
                ),
                "size_threshold": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 8192,
                        "tooltip": (
                            "Discard regions whose detected box is smaller than this "
                            "many pixels wide or tall."
                        ),
                    },
                ),
                "keep_only": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": (
                            "Keep only this many detected regions after threshold "
                            "filtering. Use 0 to keep all regions."
                        ),
                    },
                ),
                "keep_by": (
                    KEEP_BY_OPTIONS,
                    {
                        "default": "highest confidence",
                        "tooltip": (
                            "Choose how regions are ranked when Keep Only is "
                            "greater than 0."
                        ),
                    },
                ),
                "bbox_dilation": (
                    "INT",
                    {
                        "default": 0,
                        "min": -512,
                        "max": 512,
                        "step": 1,
                        "tooltip": (
                            "Grow or shrink initial detection boxes in pixels before "
                            "masks are built."
                        ),
                    },
                ),
                "sub_dilation": (
                    "INT",
                    {
                        "default": 0,
                        "min": -512,
                        "max": 512,
                        "step": 1,
                        "tooltip": (
                            "Grow or shrink the segmentation refinement mask in "
                            "pixels before it is applied to detected regions."
                        ),
                    },
                ),
                "post_dilation": (
                    "INT",
                    {
                        "default": 0,
                        "min": -512,
                        "max": 512,
                        "step": 1,
                        "tooltip": (
                            "Grow or shrink each final cropped SEG mask in pixels "
                            "after detection and refinement."
                        ),
                    },
                ),
                "crop_factor": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": (
                            "How much context to include around each detected region. "
                            "Use 0 for the full image; higher values make larger "
                            "SEG crops."
                        ),
                    },
                ),
                "sort_order": (
                    SORT_ORDER_OPTIONS,
                    {
                        "default": SORT_ORDER_OPTIONS[0],
                        "tooltip": (
                            "Order the returned SEGS before output and before the "
                            "combined mask is built."
                        ),
                    },
                ),
                "combine_segs": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "combined",
                        "label_off": "separate",
                        "tooltip": (
                            "Return one unioned SEGS region instead of separate "
                            "regions."
                        ),
                    },
                ),
            },
        }

    def detect(
        self,
        image: object,
        detector_model: UltralyticsDetectorModel,
        confidence_threshold: float,
        size_threshold: int,
        keep_only: int,
        keep_by: str,
        bbox_dilation: int,
        sub_dilation: int,
        post_dilation: int,
        crop_factor: float,
        sort_order: str,
        combine_segs: bool,
    ) -> tuple[object, object]:
        """Run detection and return SEGS plus the combined mask output."""

        image_batch = validate_image_batch(image, "SEGS detector")
        service = self.service_class()

        segs_outputs: list[object] = []
        mask_outputs: list[torch.Tensor] = []
        for single_image in iter_single_images(image_batch):
            segs = service.detect_simple(
                image=single_image,
                detector_model=detector_model,
                bbox_threshold=confidence_threshold,
                bbox_dilation=bbox_dilation,
                crop_factor=crop_factor,
                drop_size=size_threshold,
                sub_threshold=confidence_threshold,
                sub_dilation=sub_dilation,
                post_dilation=post_dilation,
            )
            finalized = finalize_detector_segs_output(
                image=single_image,
                segs=segs,
                keep_only=keep_only,
                keep_by=keep_by,
                crop_factor=crop_factor,
                sort_order=sort_order,
                combine_segs=combine_segs,
                combined_builder=type(self).combined_builder,
            )
            segs_outputs.append(finalized.segs)
            mask_outputs.append(finalized.mask)

        return segs_outputs, torch.cat(mask_outputs, dim=0)
