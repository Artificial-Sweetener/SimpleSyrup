# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for automatic SEGS from a SAM model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import torch

from ..masking.segs_mask_ops import iter_single_images, validate_image_batch
from ..runtime.progress import PhaseProgressReporter, create_comfy_phase_progress
from ..services.segs_from_sam_output_service import SEGSFromSAMOutputService


class SEGSFromSAMOutput:
    """Generate reusable unprompted SEGS from a connected SAM model."""

    service_class: ClassVar[type[SEGSFromSAMOutputService]] = SEGSFromSAMOutputService
    progress_factory: ClassVar[Callable[..., PhaseProgressReporter]] = (
        create_comfy_phase_progress
    )

    RETURN_TYPES = ("SEGS", "IMAGE")
    RETURN_NAMES = ("segs", "overlay")
    OUTPUT_IS_LIST = (True, False)
    OUTPUT_TOOLTIPS = (
        "Automatic image regions as SEGS for detailing, masking, or tiled diffusion.",
        "Source images with retained SAM regions shown as translucent colors.",
    )
    FUNCTION = "generate"
    CATEGORY = "SimpleSyrup/Detection"
    DESCRIPTION = "Creates automatic, unprompted image SEGS from a connected SAM model."
    SEARCH_ALIASES = ["sam", "automatic", "segment", "segmentation", "segs"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare inputs for automatic SAM-to-SEGS conversion."""

        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Image whose automatic regions become SEGS."},
                ),
                "sam_model": (
                    "SAM_MODEL",
                    {"tooltip": "SAM model used to find unprompted image regions."},
                ),
                "segmentation_resolution": (
                    "INT",
                    {
                        "default": 640,
                        "min": 64,
                        "max": 8192,
                        "step": 64,
                        "tooltip": (
                            "Maximum long edge in pixels used for segmentation. "
                            "Lower values run faster and omit smaller details."
                        ),
                    },
                ),
                "minimum_region_area": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 268435456,
                        "step": 1,
                        "tooltip": (
                            "Discard masks smaller than this many pixels in the "
                            "original image."
                        ),
                    },
                ),
            }
        }

    def generate(
        self,
        image: object,
        sam_model: object,
        segmentation_resolution: int = 640,
        minimum_region_area: int = 0,
    ) -> tuple[list[object], torch.Tensor]:
        """Return aligned automatic SEGS and a SAM-style overlay image batch."""

        image_batch = validate_image_batch(image, "SEGS from SAM Output")
        service = self.service_class()
        phase_progress = type(self).progress_factory(
            operation="segs_from_sam_output",
            subject=_sam_model_subject(sam_model),
            total_phases=int(image_batch.shape[0]) * 4 + 1,
        )
        outputs: list[object] = []
        overlays: list[torch.Tensor] = []
        try:
            for single_image in iter_single_images(image_batch):
                result = service.build(
                    image=single_image,
                    sam_model=sam_model,
                    segmentation_resolution=segmentation_resolution,
                    minimum_region_area=minimum_region_area,
                    phase_progress=phase_progress,
                )
                outputs.append(result.segs)
                overlays.append(result.overlay)
        except Exception:
            phase_progress.advance("failed")
            raise
        phase_progress.advance("completed")
        return outputs, torch.cat(overlays, dim=0)


def _sam_model_subject(sam_model: object) -> str:
    """Return a concise model identity for Comfy progress diagnostics."""

    model_id = getattr(sam_model, "model_id", None)
    if isinstance(model_id, str) and model_id:
        return model_id
    return type(sam_model).__name__
