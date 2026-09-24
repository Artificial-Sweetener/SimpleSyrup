# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose Impact-compatible detector facades over native detection services."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.segs import to_impact_compatible_segs
from ..runtime.ultralytics_model_adapter import UltralyticsDetectorModel
from .segs_detection_service import SegsDetectionService


@dataclass(frozen=True)
class BBoxDetectorFacade:
    """Expose a bbox detector-shaped object for existing workflows."""

    detector_model: UltralyticsDetectorModel
    detection_service: SegsDetectionService = field(
        default_factory=SegsDetectionService,
        repr=False,
        compare=False,
    )

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
        segs = self.detection_service.detect(
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
    detection_service: SegsDetectionService = field(
        default_factory=SegsDetectionService,
        repr=False,
        compare=False,
    )

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
        segs = self.detection_service.detect(
            image=image,
            detector_model=self.detector_model,
            threshold=threshold,
            dilation=dilation,
            crop_factor=crop_factor,
            drop_size=drop_size,
            prefer_segmentation=self.detector_model.supports_segmentation,
        )
        return to_impact_compatible_segs(segs)
