# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for target image resizing."""

from __future__ import annotations

from typing import Any

import torch

from ..image.resize_service import ResizeImageToTargetService


class ResizeImageToTarget:
    """Expose target image resizing controls to ComfyUI."""

    _service = ResizeImageToTargetService()

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK")
    RETURN_NAMES = ("image", "width", "height", "mask")
    OUTPUT_TOOLTIPS = (
        "Resized image batch after aspect-ratio handling and divisibility rounding.",
        "Final image width in pixels.",
        "Final image height in pixels.",
        "Resized mask aligned to the output image size.",
    )
    FUNCTION = "resize"
    CATEGORY = "SimpleSyrup/Image"
    DESCRIPTION = (
        "Resize image batches to a target size with selectable aspect handling, "
        "sampler, and CPU/GPU processor."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare ComfyUI inputs for target image resizing."""

        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Image batch to resize to the target dimensions."},
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 16384,
                        "step": 1,
                        "tooltip": (
                            "Target width in pixels before divisibility rounding."
                        ),
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 16384,
                        "step": 1,
                        "tooltip": (
                            "Target height in pixels before divisibility rounding."
                        ),
                    },
                ),
                "resize_mode": (
                    ["Stretch", "Keep AR", "Crop (Cover + Crop)", "Pad (Fit + Pad)"],
                    {
                        "default": "Keep AR",
                        "tooltip": (
                            "How aspect ratio is handled. Stretch fills exactly, crop "
                            "trims overflow, and pad fills empty space."
                        ),
                    },
                ),
                "sampling": (
                    ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                    {
                        "default": "lanczos",
                        "tooltip": (
                            "Resize filter. Sharper filters preserve detail but can "
                            "show more ringing."
                        ),
                    },
                ),
                "processor": (
                    ["cpu", "gpu"],
                    {
                        "default": "gpu",
                        "tooltip": (
                            "Processor used for resizing. GPU is usually faster; CPU "
                            "can reduce GPU memory pressure."
                        ),
                    },
                ),
                "divisible_by": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                        "tooltip": (
                            "Round final dimensions to a multiple of this value for "
                            "model or latent-size compatibility."
                        ),
                    },
                ),
                "crop_position": (
                    [
                        "center",
                        "top-left",
                        "top",
                        "top-right",
                        "left",
                        "right",
                        "bottom-left",
                        "bottom",
                        "bottom-right",
                    ],
                    {
                        "default": "center",
                        "tooltip": (
                            "Anchor used when crop mode trims overflow from the "
                            "resized image."
                        ),
                    },
                ),
                "pad_color": (
                    "STRING",
                    {
                        "default": "0, 0, 0",
                        "multiline": False,
                        "tooltip": "RGB color used to fill empty space in pad mode.",
                    },
                ),
                "max_batch_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": (
                            "Maximum images resized at once. Lower values reduce "
                            "memory use; 0 processes the full batch together."
                        ),
                    },
                ),
                "sinc_window": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": (
                            "Lanczos window size. Higher values can look sharper but "
                            "may add ringing."
                        ),
                    },
                ),
                "precision": (
                    ["fp32", "fp16", "bf16"],
                    {
                        "default": "fp32",
                        "tooltip": (
                            "Math precision for resizing. Lower precision can save "
                            "memory but may slightly change results."
                        ),
                    },
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {"tooltip": "Optional mask to resize with the image batch."},
                ),
            },
        }

    def resize(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        resize_mode: str,
        sampling: str,
        processor: str,
        divisible_by: int,
        crop_position: str,
        pad_color: str,
        max_batch_size: int,
        sinc_window: int,
        precision: str,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, int, torch.Tensor]:
        """Resize an image batch through the application service."""

        return self._service.resize(
            image=image,
            width=width,
            height=height,
            resize_mode=resize_mode,
            sampling=sampling,
            processor=processor,
            divisible_by=divisible_by,
            crop_position=crop_position,
            pad_color=pad_color,
            max_batch_size=max_batch_size,
            sinc_window=sinc_window,
            precision=precision,
            mask=mask,
        )
