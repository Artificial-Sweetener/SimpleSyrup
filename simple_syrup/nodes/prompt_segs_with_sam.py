# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for prompt-based SAM SEGS detection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import torch

from ..domain.segs import (
    KEEP_BY_OPTIONS,
    SORT_ORDER_OPTIONS,
    NativeSegs,
)
from ..masking.mask_ops import DETAIL_METHODS
from ..masking.prompt_segs_with_sam_service import PromptSEGSWithSAMService
from ..masking.segs_mask_ops import iter_single_images, validate_image_batch
from ..services.segs_output_service import (
    CombinedSegsResult,
    build_combined_segs_result,
    finalize_detector_segs_output,
)


class PromptSEGSWithSAM:
    """Expose prompt-based SAM region detection as SEGS."""

    _service = PromptSEGSWithSAMService()

    RETURN_TYPES = ("SEGS", "MASK")
    RETURN_NAMES = ("segs", "mask")
    OUTPUT_IS_LIST = (True, False)
    OUTPUT_TOOLTIPS = (
        "Prompted regions as separate or combined SEGS based on combine_segs.",
        "Combined prompted area as a standard ComfyUI mask.",
    )
    FUNCTION = "prompt"
    CATEGORY = "SimpleSyrup/Detection"
    DESCRIPTION = (
        "Finds prompt-matched regions with GroundingDINO, segments them with SAM, "
        "and returns SEGS plus a combined mask."
    )
    SEARCH_ALIASES = ["sam", "groundingdino", "prompt", "segs"]

    combined_builder: ClassVar[
        Callable[[object, NativeSegs, float], CombinedSegsResult]
    ] = build_combined_segs_result

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        """Declare deterministic ComfyUI inputs without model IO."""

        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Image to search with the text prompt."},
                ),
                "sam_model": (
                    "SAM_MODEL",
                    {
                        "tooltip": (
                            "SAM model used to create masks for the prompted boxes."
                        )
                    },
                ),
                "grounding_dino_model": (
                    "GROUNDING_DINO_MODEL,DINO_MODEL",
                    {
                        "tooltip": (
                            "GroundingDINO model used to find boxes that match the "
                            "prompt."
                        )
                    },
                ),
                "positive_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Text describing the regions to detect.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Text describing areas to subtract from the detected "
                            "regions."
                        ),
                    },
                ),
                "confidence_threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Minimum GroundingDINO confidence required to create a "
                            "SAM prompt box."
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
                            "Discard final regions whose mask bounds are smaller "
                            "than this many pixels wide or tall."
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
                            "Grow or shrink prompt boxes in pixels before sending "
                            "them to SAM."
                        ),
                    },
                ),
                "mask_dilation": (
                    "INT",
                    {
                        "default": 0,
                        "min": -512,
                        "max": 512,
                        "step": 1,
                        "tooltip": (
                            "Grow or shrink each final region mask in pixels after "
                            "prompt subtraction."
                        ),
                    },
                ),
                "detail_method": (
                    list(DETAIL_METHODS),
                    {
                        "default": "GuidedFilter",
                        "tooltip": "Method used to refine mask edges.",
                    },
                ),
                "detail_erode": (
                    "INT",
                    {
                        "default": 6,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "tooltip": (
                            "Pixels trimmed inside the mask edge during refinement. "
                            "Higher values pull the refined edge inward."
                        ),
                    },
                ),
                "detail_dilate": (
                    "INT",
                    {
                        "default": 6,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "tooltip": (
                            "Pixels added outside the mask edge during refinement. "
                            "Higher values give refinement more outside context."
                        ),
                    },
                ),
                "black_point": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 0.98,
                        "step": 0.01,
                        "tooltip": "Mask values at or below this become black.",
                    },
                ),
                "white_point": (
                    "FLOAT",
                    {
                        "default": 0.99,
                        "min": 0.02,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Mask values at or above this become white.",
                    },
                ),
                "refine_mask": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "enabled",
                        "label_off": "disabled",
                        "tooltip": "Enable edge refinement for the final masks.",
                    },
                ),
                "mask_refinement_max_size": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 1,
                        "max": 16384,
                        "step": 64,
                        "tooltip": (
                            "Maximum working size for mask refinement. Larger "
                            "images are refined at a bounded size for performance."
                        ),
                    },
                ),
                "execution_device": (
                    ["auto", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Device for model-backed refinement. Auto uses normal "
                            "ComfyUI placement; CPU avoids GPU memory pressure."
                        ),
                    },
                ),
                "crop_factor": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 1.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": (
                            "How much context to include around each prompted "
                            "region. Higher values make larger SEG crops."
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
            "optional": {
                "vitmatte_model": (
                    "VITMATTE_MODEL",
                    {
                        "tooltip": (
                            "Optional ViTMatte model used when detail_method is "
                            "VITMatte."
                        )
                    },
                ),
            },
        }

    def prompt(
        self,
        image: object,
        sam_model: object,
        grounding_dino_model: object,
        positive_prompt: str,
        negative_prompt: str,
        confidence_threshold: float,
        size_threshold: int,
        keep_only: int,
        keep_by: str,
        bbox_dilation: int,
        mask_dilation: int,
        detail_method: str,
        detail_erode: int,
        detail_dilate: int,
        black_point: float,
        white_point: float,
        refine_mask: bool,
        mask_refinement_max_size: int,
        execution_device: str,
        crop_factor: float,
        sort_order: str,
        combine_segs: bool,
        vitmatte_model: object | None = None,
    ) -> tuple[object, object]:
        """Prompt regions and return SEGS plus the combined mask output."""

        image_batch = validate_image_batch(image, "Prompt SEGS w/ SAM")

        segs_outputs: list[object] = []
        mask_outputs: list[torch.Tensor] = []
        for single_image in iter_single_images(image_batch):
            segs = self._service.prompt(
                image=single_image,
                sam_model=sam_model,
                grounding_dino_model=grounding_dino_model,
                vitmatte_model=vitmatte_model,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                confidence_threshold=confidence_threshold,
                size_threshold=size_threshold,
                bbox_dilation=bbox_dilation,
                mask_dilation=mask_dilation,
                detail_method=detail_method,
                detail_erode=detail_erode,
                detail_dilate=detail_dilate,
                black_point=black_point,
                white_point=white_point,
                refine_mask=refine_mask,
                mask_refinement_max_size=mask_refinement_max_size,
                execution_device=execution_device,
                crop_factor=crop_factor,
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
