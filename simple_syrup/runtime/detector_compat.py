# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Impact-style detector facades backed by native SimpleSyrup services."""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.segs import to_impact_compatible_segs
from .ultralytics_loader import UltralyticsDetectorModel


@dataclass(frozen=True)
class BBoxDetectorFacade:
    """Expose a bbox detector-shaped object for existing workflows."""

    detector_model: UltralyticsDetectorModel

    def detect(
        self,
        image: object,
        threshold: float,
        dilation: int,
        crop_factor: float,
        drop_size: int = 1,
        detailer_hook: object | None = None,
    ) -> object:
        """Detect rectangular SEGS through the native detection service."""

        del detailer_hook
        from ..services.segs_detection_service import SegsDetectionService

        segs = SegsDetectionService().detect(
            image=image,
            detector_model=self.detector_model,
            threshold=threshold,
            dilation=dilation,
            crop_factor=crop_factor,
            drop_size=drop_size,
            prefer_segmentation=False,
        )
        return to_impact_compatible_segs(segs)


@dataclass(frozen=True)
class SegmDetectorFacade:
    """Expose a segmentation detector-shaped object with bbox fallback."""

    detector_model: UltralyticsDetectorModel
    bbox_detector: BBoxDetectorFacade

    def detect(
        self,
        image: object,
        threshold: float,
        dilation: int,
        crop_factor: float,
        drop_size: int = 1,
        detailer_hook: object | None = None,
    ) -> object:
        """Detect segmentation SEGS when available, otherwise rectangular SEGS."""

        del detailer_hook
        from ..services.segs_detection_service import SegsDetectionService

        segs = SegsDetectionService().detect(
            image=image,
            detector_model=self.detector_model,
            threshold=threshold,
            dilation=dilation,
            crop_factor=crop_factor,
            drop_size=drop_size,
            prefer_segmentation=self.detector_model.supports_segmentation,
        )
        return to_impact_compatible_segs(segs)
