# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for loading Ultralytics models."""

from __future__ import annotations

from typing import Any, ClassVar

from ..runtime.ultralytics_loader import UltralyticsLoaderService


class LoadUltralyticsModel:
    """Load one Ultralytics model and expose native and compatibility outputs."""

    RETURN_TYPES = ("DETECTOR_MODEL", "BBOX_DETECTOR", "SEGM_DETECTOR")
    RETURN_NAMES = ("detector_model", "bbox_detector", "segm_detector")
    OUTPUT_TOOLTIPS = (
        "Detector model for SimpleSyrup SEGS detection nodes.",
        "Bounding-box detector output for nodes that expect a bbox detector.",
        "Segmentation detector output for nodes that expect a mask-capable detector.",
    )
    FUNCTION = "load"
    CATEGORY = "SimpleSyrup/Detection"
    DESCRIPTION = "Loads an Ultralytics model for detection and SEGS workflows."
    SEARCH_ALIASES = ["ultralytics", "yolo", "detector"]

    service_class: ClassVar[type[UltralyticsLoaderService]] = UltralyticsLoaderService

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare ComfyUI inputs for Ultralytics model loading."""

        choices = cls.service_class().model_choices()
        return {
            "required": {
                "model_name": (
                    choices,
                    {
                        "default": choices[0],
                        "tooltip": (
                            "Ultralytics model file in the ComfyUI models folder."
                        ),
                    },
                )
            }
        }

    def load(self, model_name: str) -> tuple[object, object, object]:
        """Load the selected detector and paired compatibility facades."""

        loaded = self.service_class().load(model_name)
        return loaded.detector_model, loaded.bbox_detector, loaded.segm_detector
