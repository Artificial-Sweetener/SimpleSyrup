# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure mask composition and refinement helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

DETAIL_METHODS = (
    "GuidedFilter",
    "PyMatting",
    "VITMatte",
)


@dataclass(frozen=True)
class MaskRefinementSettings:
    """Controls for standalone prompt mask post-processing."""

    detail_method: str
    detail_erode: int
    detail_dilate: int
    black_point: float
    white_point: float
    process_detail: bool
    execution_device: str
    max_size_pixels: int


def zero_mask_like_image(image: torch.Tensor) -> torch.Tensor:
    """Return an all-zero BHW mask matching a BHWC image tensor."""

    batch_size, height, width, _channels = image.shape
    return torch.zeros(
        (int(batch_size), int(height), int(width)),
        dtype=torch.float32,
        device=image.device,
    )


def compose_prompt_masks(
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return `positive_mask - negative_mask` clamped to ComfyUI mask range."""

    positive = positive_mask.float()
    if negative_mask is None:
        return positive.clamp(0.0, 1.0)
    return (positive - negative_mask.float()).clamp(0.0, 1.0)


def refine_prompt_mask(
    mask: torch.Tensor,
    image: torch.Tensor,
    settings: MaskRefinementSettings,
) -> torch.Tensor:
    """Apply level remapping and optional edge-detail refinement."""

    _validate_refinement_settings(settings)
    output_device = mask.device
    work_device = _resolve_refinement_device(settings.execution_device)
    working_mask = mask.to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)
    working_image = image.to(device=work_device, dtype=torch.float32).clamp(0.0, 1.0)

    if settings.process_detail:
        working_mask = _refine_detail(working_mask, working_image, settings)

    remapped = histogram_remap(working_mask, settings.black_point, settings.white_point)
    return remapped.to(device=output_device, dtype=torch.float32).clamp(0.0, 1.0)


def histogram_remap(
    mask: torch.Tensor,
    black_point: float,
    white_point: float,
) -> torch.Tensor:
    """Remap mask values between black and white points into `[0, 1]`."""

    if white_point <= black_point:
        raise ValueError("white_point must be greater than black_point.")
    return ((mask.float() - black_point) / (white_point - black_point)).clamp(0.0, 1.0)


def _validate_refinement_settings(settings: MaskRefinementSettings) -> None:
    """Reject invalid mask refinement settings."""

    if settings.detail_method not in DETAIL_METHODS:
        valid = ", ".join(DETAIL_METHODS)
        raise ValueError(f"detail_method must be one of: {valid}.")
    if settings.detail_erode < 0:
        raise ValueError("detail_erode must be greater than or equal to 0.")
    if settings.detail_dilate < 0:
        raise ValueError("detail_dilate must be greater than or equal to 0.")
    if not 0.0 <= settings.black_point < settings.white_point <= 1.0:
        raise ValueError(
            "black_point and white_point must satisfy 0 <= black < white <= 1."
        )
    if settings.execution_device not in ("auto", "cpu"):
        raise ValueError("execution_device must be 'auto' or 'cpu'.")
    if settings.max_size_pixels < 1:
        raise ValueError("max_size_pixels must be at least 1.")


def _resolve_refinement_device(device: str) -> torch.device:
    """Return the requested torch device or fail clearly."""

    return torch.device("cpu")


def _refine_detail(
    mask: torch.Tensor,
    image: torch.Tensor,
    settings: MaskRefinementSettings,
) -> torch.Tensor:
    """Refine mask edges using a bounded-size native torch implementation."""

    if settings.detail_method == "VITMatte":
        raise ValueError(
            "detail_method 'VITMatte' requires a connected VITMATTE_MODEL when "
            "refine_mask is enabled. Connect ViTMatte Model Loader or choose "
            "GuidedFilter."
        )

    height = int(mask.shape[-2])
    width = int(mask.shape[-1])
    max_dimension = max(height, width)
    if max_dimension > settings.max_size_pixels:
        scale = settings.max_size_pixels / float(max_dimension)
        scaled_height = max(1, int(round(height * scale)))
        scaled_width = max(1, int(round(width * scale)))
        scaled_settings = replace(
            settings,
            detail_erode=max(1, int(round(settings.detail_erode * scale))),
            detail_dilate=max(1, int(round(settings.detail_dilate * scale))),
        )
        small_mask = _resize_mask(mask, scaled_height, scaled_width)
        small_image = _resize_image(image, scaled_height, scaled_width)
        refined = _refine_detail_at_size(small_mask, small_image, scaled_settings)
        return _resize_mask(refined, height, width)

    return _refine_detail_at_size(mask, image, settings)


def _refine_detail_at_size(
    mask: torch.Tensor,
    image: torch.Tensor,
    settings: MaskRefinementSettings,
) -> torch.Tensor:
    """Run one detail refinement pass at the current tensor size."""

    mask_bchw = mask.unsqueeze(1)
    edge_band = (
        _dilate(mask_bchw, settings.detail_dilate)
        - _erode(mask_bchw, settings.detail_erode)
    ).clamp(0.0, 1.0)

    if settings.detail_method == "GuidedFilter":
        smoothed = _guided_filter_mask(image, mask, _detail_radius(settings)).unsqueeze(
            1
        )
    else:
        smoothed = _box_blur(mask_bchw, _detail_radius(settings))

    refined = mask_bchw * (1.0 - edge_band) + smoothed * edge_band
    return refined.squeeze(1).clamp(0.0, 1.0)


def _detail_radius(settings: MaskRefinementSettings) -> int:
    """Return a method-specific smoothing radius."""

    detail_range = settings.detail_erode + settings.detail_dilate
    if settings.detail_method == "GuidedFilter":
        return max(1, detail_range // 6 + 1)
    if settings.detail_method == "PyMatting":
        return max(1, detail_range // 8 + 1)
    return max(1, detail_range // 4 + 1)


def _guided_filter_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    radius: int,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Apply a grayscale guided filter to a BHW mask using a BHWC image."""

    guidance = _grayscale_bchw(image)
    target = mask.unsqueeze(1)
    mean_i = _box_blur(guidance, radius)
    mean_p = _box_blur(target, radius)
    corr_i = _box_blur(guidance * guidance, radius)
    corr_ip = _box_blur(guidance * target, radius)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + epsilon)
    b = mean_p - a * mean_i
    mean_a = _box_blur(a, radius)
    mean_b = _box_blur(b, radius)
    return (mean_a * guidance + mean_b).squeeze(1).clamp(0.0, 1.0)


def _grayscale_bchw(image: torch.Tensor) -> torch.Tensor:
    """Convert a BHWC image tensor to B1HW grayscale guidance."""

    channels = int(image.shape[-1])
    if channels >= 3:
        weights = torch.tensor(
            [0.299, 0.587, 0.114],
            dtype=image.dtype,
            device=image.device,
        )
        gray = (image[..., :3] * weights).sum(dim=-1)
    else:
        gray = image[..., 0]
    return gray.unsqueeze(1)


def _box_blur(samples: torch.Tensor, radius: int) -> torch.Tensor:
    """Blur BCHW samples with edge-aware average pooling."""

    if radius <= 0:
        return samples
    kernel_size = radius * 2 + 1
    return F.avg_pool2d(
        samples,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
        count_include_pad=False,
    )


def _dilate(samples: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate BCHW mask samples."""

    if radius <= 0:
        return samples
    kernel_size = radius * 2 + 1
    padded = F.pad(samples, (radius, radius, radius, radius), value=0.0)
    return F.max_pool2d(padded, kernel_size=kernel_size, stride=1)


def _erode(samples: torch.Tensor, radius: int) -> torch.Tensor:
    """Erode BCHW mask samples."""

    if radius <= 0:
        return samples
    kernel_size = radius * 2 + 1
    padded = F.pad(samples, (radius, radius, radius, radius), value=0.0)
    return -F.max_pool2d(-padded, kernel_size=kernel_size, stride=1)


def _resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize a BHW mask tensor."""

    return F.interpolate(
        mask.unsqueeze(1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def _resize_image(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize a BHWC image tensor."""

    resized = F.interpolate(
        image.movedim(-1, 1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.movedim(1, -1)
