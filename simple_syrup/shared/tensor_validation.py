# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tensor shape validation for ComfyUI image and mask inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ImageTensorShape:
    """Validated ComfyUI IMAGE tensor dimensions."""

    batch_size: int
    height: int
    width: int
    channels: int


@dataclass(frozen=True)
class MaskTensorShape:
    """Validated ComfyUI MASK tensor dimensions."""

    batch_size: int
    height: int
    width: int


def validate_image_tensor(image: object) -> ImageTensorShape:
    """Validate a BHWC floating-point ComfyUI image tensor."""

    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor with shape (B, H, W, C).")
    if image.ndim != 4:
        raise ValueError(
            f"image must have shape (B, H, W, C), got {tuple(image.shape)}."
        )
    if not image.is_floating_point():
        raise TypeError("image must be a floating-point tensor with values in [0, 1].")

    batch_size, height, width, channels = (int(value) for value in image.shape)
    if batch_size <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            "image dimensions must be positive, got "
            f"batch={batch_size}, height={height}, width={width}."
        )
    if channels not in (1, 3, 4):
        raise ValueError(
            f"Unsupported image channel count {channels}. Expected 1, 3, or 4."
        )

    return ImageTensorShape(
        batch_size=batch_size,
        height=height,
        width=width,
        channels=channels,
    )


def validate_mask_tensor(mask: object, batch_size: int) -> MaskTensorShape:
    """Validate a BHW floating-point ComfyUI mask tensor."""

    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor with shape (B, H, W).")
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B, H, W), got {tuple(mask.shape)}.")
    if not mask.is_floating_point():
        raise TypeError("mask must be a floating-point tensor with values in [0, 1].")

    mask_batch, height, width = (int(value) for value in mask.shape)
    if mask_batch != batch_size:
        raise ValueError(
            f"mask batch size {mask_batch} must match image batch size {batch_size}."
        )
    if height <= 0 or width <= 0:
        raise ValueError(
            f"mask dimensions must be positive, got height={height}, width={width}."
        )

    return MaskTensorShape(batch_size=mask_batch, height=height, width=width)
