# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node for converting masks into image-associated SEGS."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from ..domain.segs import SORT_ORDER_OPTIONS, NativeSegs
from ..masking.segs_mask_ops import iter_single_images, validate_image_batch
from ..services.mask_to_segs_service import MaskToSEGSService
from ..services.segs_output_service import (
    CombinedSegsResult,
    build_combined_segs_result,
    finalize_detector_segs_output,
)

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class MaskToSEGSV3(_ComfyNodeBase):
    """Expose image-associated mask-to-SEGS conversion through Comfy v3."""

    service_class: ClassVar[type[MaskToSEGSService]] = MaskToSEGSService
    combined_builder: ClassVar[
        Callable[[object, NativeSegs, float], CombinedSegsResult]
    ] = build_combined_segs_result

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the Mask to SEGS v3 schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.MaskToSEGS",
            display_name="Mask to SEGS",
            category="SimpleSyrup/Detection",
            description=(
                "Converts an existing mask into image-associated SEGS for detail "
                "and regional workflows."
            ),
            search_aliases=["mask", "segs", "region", "detect", "segmentation"],
            inputs=[
                _comfy_io.Image.Input(
                    "image",
                    tooltip=(
                        "Source image used to create cropped image data for each SEG."
                    ),
                ),
                _comfy_io.Mask.Input(
                    "mask",
                    tooltip="Mask whose active regions become SEGS.",
                ),
                _comfy_io.Float.Input(
                    "mask_threshold",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Mask value required for a pixel to count as active.",
                ),
                _comfy_io.Int.Input(
                    "size_threshold",
                    default=10,
                    min=1,
                    max=8192,
                    step=1,
                    tooltip=(
                        "Discard regions smaller than this many pixels wide or tall."
                    ),
                ),
                _comfy_io.Int.Input(
                    "keep_only",
                    default=0,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip="Keep only this many largest regions. Use 0 to keep all.",
                ),
                _comfy_io.Int.Input(
                    "mask_dilation",
                    default=0,
                    min=-512,
                    max=512,
                    step=1,
                    tooltip=(
                        "Grow or shrink the source mask in pixels before regions "
                        "are found."
                    ),
                ),
                _comfy_io.Int.Input(
                    "post_dilation",
                    default=0,
                    min=-512,
                    max=512,
                    step=1,
                    tooltip=(
                        "Grow or shrink each final cropped SEG mask in pixels "
                        "after cropping."
                    ),
                ),
                _comfy_io.Float.Input(
                    "crop_factor",
                    default=3.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    tooltip=(
                        "Context around each region. Use 0 for the full image; "
                        "higher values make larger crops."
                    ),
                ),
                _comfy_io.Combo.Input(
                    "sort_order",
                    options=list(SORT_ORDER_OPTIONS),
                    default=SORT_ORDER_OPTIONS[0],
                    tooltip="Order separate SEGS before output and combining.",
                ),
                _comfy_io.Boolean.Input(
                    "combine_segs",
                    default=False,
                    label_on="combined",
                    label_off="separate",
                    tooltip="Return one unioned SEG instead of separate mask regions.",
                ),
                _comfy_io.String.Input(
                    "label",
                    multiline=False,
                    default="mask",
                    tooltip="Label stored on each extracted SEG.",
                ),
            ],
            outputs=[
                _comfy_io.SEGS.Output(
                    "segs",
                    tooltip="Image-associated SEGS created from the mask.",
                    is_output_list=True,
                ),
                _comfy_io.Mask.Output(
                    "mask",
                    tooltip="Union of retained SEGS as a standard ComfyUI mask.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: object,
        mask: object,
        mask_threshold: float,
        size_threshold: int,
        keep_only: int,
        mask_dilation: int,
        post_dilation: int,
        crop_factor: float,
        sort_order: str,
        combine_segs: bool,
        label: str,
    ) -> tuple[object, torch.Tensor]:
        """Convert a mask batch into SEGS outputs and a combined mask batch."""

        image_batch = validate_image_batch(image, "Mask to SEGS")
        mask_batch = _validate_mask_batch(mask, image_batch)
        service = cls.service_class()

        segs_outputs: list[object] = []
        mask_outputs: list[torch.Tensor] = []
        for index, single_image in enumerate(iter_single_images(image_batch)):
            mask_index = index if int(mask_batch.shape[0]) > 1 else 0
            segs = service.build(
                image=single_image,
                mask=mask_batch[mask_index : mask_index + 1],
                mask_threshold=mask_threshold,
                size_threshold=size_threshold,
                mask_dilation=mask_dilation,
                post_dilation=post_dilation,
                crop_factor=crop_factor,
                label=label,
            )
            finalized = finalize_detector_segs_output(
                image=single_image,
                segs=segs,
                keep_only=keep_only,
                keep_by="largest size",
                crop_factor=crop_factor,
                sort_order=sort_order,
                combine_segs=combine_segs,
                combined_builder=cls.combined_builder,
            )
            segs_outputs.append(finalized.segs)
            mask_outputs.append(finalized.mask)

        return segs_outputs, torch.cat(mask_outputs, dim=0)


def _validate_mask_batch(mask: object, image_batch: torch.Tensor) -> torch.Tensor:
    """Return a BHW mask batch compatible with an image batch."""

    if not isinstance(mask, torch.Tensor):
        raise TypeError("Mask to SEGS requires a torch MASK tensor.")
    working = mask.float()
    if working.ndim == 2:
        working = working.unsqueeze(0)
    if working.ndim != 3:
        raise ValueError("Mask to SEGS requires an HW or BHW mask tensor.")

    image_batch_size = int(image_batch.shape[0])
    mask_batch_size = int(working.shape[0])
    if mask_batch_size not in (1, image_batch_size):
        raise ValueError(
            "Mask to SEGS requires mask batch size to be 1 or match image batch "
            f"size; mask batch is {mask_batch_size}, image batch is "
            f"{image_batch_size}."
        )

    image_height = int(image_batch.shape[1])
    image_width = int(image_batch.shape[2])
    mask_height = int(working.shape[1])
    mask_width = int(working.shape[2])
    if mask_height != image_height or mask_width != image_width:
        raise ValueError(
            "Mask to SEGS requires mask dimensions to match image dimensions; "
            f"mask is {mask_height}x{mask_width}, image is "
            f"{image_height}x{image_width}."
        )
    return working.clamp(0.0, 1.0)
