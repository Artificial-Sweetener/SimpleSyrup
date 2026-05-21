# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application service for prompt-based SAM SEGS construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..domain.segs import (
    BoundingBox,
    NativeSegs,
    Segment,
    sort_segs,
)
from ..masking.mask_ops import MaskRefinementSettings, refine_prompt_mask
from ..masking.segs_mask_ops import (
    crop_image,
    crop_mask,
    crop_region_for_bbox,
    dilate_mask,
    normalize_mask,
    validate_single_image,
)
from ..runtime.sam_segmenter import SAMBoxSegmenter, SAMModelSegmenter
from ..runtime.text_box_detector import (
    GroundingDINOTextBoxDetector,
    TextBoxDetection,
    TextBoxDetector,
)
from ..runtime.vitmatte_refiner import MaskDetailRefiner, ViTMatteRefiner


@dataclass(frozen=True)
class PromptSegsRuntime:
    """Runtime dependencies used to detect prompt boxes and segment masks."""

    detector: TextBoxDetector
    segmenter: SAMBoxSegmenter


@dataclass(frozen=True)
class PromptSegsSettings:
    """Validated controls for prompt-to-SEGS construction."""

    positive_prompt: str
    negative_prompt: str
    confidence_threshold: float
    size_threshold: int
    bbox_dilation: int
    mask_dilation: int
    crop_factor: float
    sort_order: str
    refinement: MaskRefinementSettings


class PromptSEGSWithSAMService:
    """Validate inputs and build prompt-derived SEGS with SAM masks."""

    def __init__(
        self,
        runtime: PromptSegsRuntime | None = None,
        vitmatte_refiner: MaskDetailRefiner | None = None,
    ) -> None:
        """Create the service with injectable runtime dependencies."""

        self._runtime = runtime or PromptSegsRuntime(
            detector=GroundingDINOTextBoxDetector(),
            segmenter=SAMModelSegmenter(),
        )
        self._vitmatte_refiner = vitmatte_refiner or ViTMatteRefiner()

    def prompt(
        self,
        image: object,
        sam_model: object,
        grounding_dino_model: object,
        vitmatte_model: object | None,
        positive_prompt: str,
        negative_prompt: str,
        confidence_threshold: float,
        size_threshold: int,
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
    ) -> NativeSegs:
        """Return sorted native SEGS for a text-prompted SAM detection."""

        image_tensor = validate_single_image(image, "Prompt SEGS w/ SAM")
        settings = self._validate_settings(
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
            sort_order=sort_order,
        )

        sample = image_tensor[0]
        height = int(sample.shape[0])
        width = int(sample.shape[1])
        positive_regions = self._prompt_regions(
            image=sample,
            sam_model=sam_model,
            grounding_dino_model=grounding_dino_model,
            prompt=settings.positive_prompt,
            settings=settings,
            execution_device=execution_device,
        )
        negative_mask = self._negative_mask(
            image=sample,
            sam_model=sam_model,
            grounding_dino_model=grounding_dino_model,
            settings=settings,
            execution_device=execution_device,
            height=height,
            width=width,
        )

        segments: list[Segment] = []
        for detection, mask in positive_regions:
            final_mask = (mask - negative_mask).clamp(0.0, 1.0)
            final_mask = dilate_mask(final_mask, settings.mask_dilation)
            final_mask = self._refine_mask(
                image_tensor=image_tensor,
                mask=final_mask,
                settings=settings,
                vitmatte_model=vitmatte_model,
            )
            bbox = _bbox_from_mask(final_mask)
            if bbox is None:
                continue
            if (
                bbox.width < settings.size_threshold
                or bbox.height < settings.size_threshold
            ):
                continue
            crop_region = crop_region_for_bbox(
                bbox,
                image_height=height,
                image_width=width,
                crop_factor=settings.crop_factor,
            )
            segments.append(
                Segment(
                    cropped_image=crop_image(image_tensor, crop_region)
                    .detach()
                    .clone(),
                    cropped_mask=crop_mask(final_mask, crop_region).detach().clone(),
                    confidence=detection.confidence,
                    crop_region=crop_region,
                    bbox=bbox,
                    label=settings.positive_prompt,
                )
            )

        return sort_segs(((height, width), tuple(segments)), settings.sort_order)

    def _prompt_regions(
        self,
        image: torch.Tensor,
        sam_model: object,
        grounding_dino_model: object,
        prompt: str,
        settings: PromptSegsSettings,
        execution_device: str,
    ) -> tuple[tuple[TextBoxDetection, torch.Tensor], ...]:
        """Return detected prompt boxes with their SAM masks."""

        detections = self._runtime.detector.detect(
            grounding_dino_model,
            image,
            prompt,
            settings.confidence_threshold,
            execution_device,
        )
        if not detections:
            return ()

        height = int(image.shape[0])
        width = int(image.shape[1])
        expanded_boxes = tuple(
            _expand_bbox(detection.bbox, settings.bbox_dilation, height, width)
            for detection in detections
        )
        masks = self._runtime.segmenter.segment_boxes(
            sam_model,
            image,
            _boxes_tensor(expanded_boxes),
            settings.confidence_threshold,
            execution_device,
        )
        if len(masks) != len(detections):
            raise ValueError(
                "Prompt SEGS w/ SAM runtime returned a mask count that does not "
                "match prompt detections."
            )
        return tuple(
            (
                detection,
                normalize_mask(mask, height, width),
            )
            for detection, mask in zip(detections, masks, strict=True)
        )

    def _negative_mask(
        self,
        image: torch.Tensor,
        sam_model: object,
        grounding_dino_model: object,
        settings: PromptSegsSettings,
        execution_device: str,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Return one combined negative prompt mask."""

        if not settings.negative_prompt:
            return torch.zeros((height, width), dtype=torch.float32)

        negative_regions = self._prompt_regions(
            image=image,
            sam_model=sam_model,
            grounding_dino_model=grounding_dino_model,
            prompt=settings.negative_prompt,
            settings=settings,
            execution_device=execution_device,
        )
        if not negative_regions:
            return torch.zeros((height, width), dtype=torch.float32)
        return torch.stack([mask for _detection, mask in negative_regions]).amax(dim=0)

    def _refine_mask(
        self,
        image_tensor: torch.Tensor,
        mask: torch.Tensor,
        settings: PromptSegsSettings,
        vitmatte_model: object | None,
    ) -> torch.Tensor:
        """Apply configured mask refinement to one full-image mask."""

        mask_batch = mask.unsqueeze(0)
        if (
            settings.refinement.process_detail
            and settings.refinement.detail_method == "VITMatte"
        ):
            vitmatte_refined = self._vitmatte_refiner.refine(
                image_tensor,
                mask_batch,
                settings.refinement,
                vitmatte_model,
            )
            remap_settings = MaskRefinementSettings(
                detail_method="GuidedFilter",
                detail_erode=0,
                detail_dilate=0,
                black_point=settings.refinement.black_point,
                white_point=settings.refinement.white_point,
                process_detail=False,
                execution_device=settings.refinement.execution_device,
                max_size_pixels=settings.refinement.max_size_pixels,
            )
            return refine_prompt_mask(
                vitmatte_refined,
                image_tensor,
                remap_settings,
            )[0]

        return refine_prompt_mask(mask_batch, image_tensor, settings.refinement)[0]

    def _validate_settings(
        self,
        positive_prompt: str,
        negative_prompt: str,
        confidence_threshold: float,
        size_threshold: int,
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
    ) -> PromptSegsSettings:
        """Validate public node settings and return normalized values."""

        positive_text = positive_prompt.strip()
        if not positive_text:
            raise ValueError("positive_prompt is required for Prompt SEGS w/ SAM.")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if size_threshold < 1:
            raise ValueError("size_threshold must be at least 1.")
        if crop_factor < 1.0:
            raise ValueError("crop_factor must be at least 1.0.")
        if mask_refinement_max_size < 1:
            raise ValueError("mask_refinement_max_size must be at least 1.")

        refinement = MaskRefinementSettings(
            detail_method=detail_method,
            detail_erode=int(detail_erode),
            detail_dilate=int(detail_dilate),
            black_point=float(black_point),
            white_point=float(white_point),
            process_detail=bool(refine_mask),
            execution_device=execution_device,
            max_size_pixels=int(mask_refinement_max_size),
        )
        refine_prompt_mask(
            torch.zeros((1, 1, 1), dtype=torch.float32),
            torch.zeros((1, 1, 1, 3), dtype=torch.float32),
            MaskRefinementSettings(
                detail_method=refinement.detail_method,
                detail_erode=refinement.detail_erode,
                detail_dilate=refinement.detail_dilate,
                black_point=refinement.black_point,
                white_point=refinement.white_point,
                process_detail=False,
                execution_device=refinement.execution_device,
                max_size_pixels=refinement.max_size_pixels,
            ),
        )

        return PromptSegsSettings(
            positive_prompt=positive_text,
            negative_prompt=negative_prompt.strip(),
            confidence_threshold=float(confidence_threshold),
            size_threshold=int(size_threshold),
            bbox_dilation=int(bbox_dilation),
            mask_dilation=int(mask_dilation),
            crop_factor=float(crop_factor),
            sort_order=sort_order,
            refinement=refinement,
        )


def _expand_bbox(
    bbox: BoundingBox,
    dilation: int,
    image_height: int,
    image_width: int,
) -> BoundingBox:
    """Expand a bbox by a signed pixel radius and clamp it to image bounds."""

    left = max(0, min(image_width, bbox.left - dilation))
    top = max(0, min(image_height, bbox.top - dilation))
    right = max(0, min(image_width, bbox.right + dilation))
    bottom = max(0, min(image_height, bbox.bottom + dilation))
    if right <= left or bottom <= top:
        return bbox
    return BoundingBox(left, top, right, bottom)


def _boxes_tensor(boxes: tuple[BoundingBox, ...]) -> torch.Tensor:
    """Convert bbox values to an XYXY tensor for SAM runtimes."""

    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32)
    return torch.tensor(
        [[box.left, box.top, box.right, box.bottom] for box in boxes],
        dtype=torch.float32,
    )


def _bbox_from_mask(mask: torch.Tensor) -> BoundingBox | None:
    """Return the tight bbox around a nonzero HW mask."""

    y_coords, x_coords = torch.where(mask > 0)
    if y_coords.numel() == 0 or x_coords.numel() == 0:
        return None
    left = int(torch.min(x_coords).item())
    top = int(torch.min(y_coords).item())
    right = int(torch.max(x_coords).item()) + 1
    bottom = int(torch.max(y_coords).item()) + 1
    return BoundingBox(left, top, right, bottom)
