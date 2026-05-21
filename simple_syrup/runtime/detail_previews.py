# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Full-context preview composition for detailer sampling callbacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, TypeAlias, cast

import torch
from PIL import Image, ImageDraw

from ..domain.segs import CropRegion

DETAIL_PREVIEW_WASH_OPACITY = 0.55
DETAIL_PREVIEW_OUTLINE_RGB = (255, 0, 0)
DETAIL_PREVIEW_OUTLINE_WIDTH = 1

CropBox: TypeAlias = tuple[int, int, int, int]
PreviewBytes: TypeAlias = tuple[str, Image.Image, int]
DetailPreviewCallback: TypeAlias = Callable[
    [int, torch.Tensor, torch.Tensor, int], None
]


class LatentPreviewer(Protocol):
    """Decode latent tensors into preview images."""

    def decode_latent_to_preview(self, x0: torch.Tensor) -> Image.Image:
        """Decode a denoised latent tensor to a PIL preview image."""


class ProgressBar(Protocol):
    """Progress sink compatible with ComfyUI's progress bar."""

    def update_absolute(
        self,
        value: int,
        total: int,
        preview: PreviewBytes | None = None,
    ) -> None:
        """Update absolute progress with optional preview image bytes."""


class ProgressBarFactory(Protocol):
    """Construct a ComfyUI progress bar for a sampling pass."""

    def __call__(self, steps: int) -> ProgressBar:
        """Return a progress bar for the provided step count."""


class LatentPreviewModule(Protocol):
    """ComfyUI latent preview module surface used by detail previews."""

    MAX_PREVIEW_RESOLUTION: int

    def get_previewer(
        self,
        device: object,
        latent_format: object,
    ) -> LatentPreviewer | None:
        """Return a latent previewer when ComfyUI previews are enabled."""


class ComfyUtilsModule(Protocol):
    """ComfyUI utility module surface used by detail previews."""

    ProgressBar: ProgressBarFactory


class InnerModelPreviewSource(Protocol):
    """Inner ComfyUI model fields needed for previewer lookup."""

    latent_format: object


class ModelPreviewSource(Protocol):
    """ComfyUI model fields needed for previewer lookup."""

    load_device: object
    model: InnerModelPreviewSource


@dataclass(frozen=True)
class DetailPreviewContext:
    """Describe preview composition for one active detailer work area."""

    image: torch.Tensor
    work_region: CropRegion
    work_mask: torch.Tensor
    sampled_region: CropRegion | None = None


@dataclass(frozen=True)
class DetailPreviewGeometry:
    """Map source work coordinates into preview image coordinates.

    `crop_box` uses PIL's exclusive right/bottom box convention for resizing and
    pasting. `outline_box` uses inclusive right/bottom coordinates for
    `ImageDraw.rectangle`.
    """

    source_size: tuple[int, int]
    preview_size: tuple[int, int]
    crop_box: CropBox
    outline_box: CropBox


@dataclass(frozen=True)
class DetailPreviewCompositor:
    """Compose active detail previews into a washed full-image context."""

    geometry: DetailPreviewGeometry
    washed_background: Image.Image
    detail_alpha_mask: Image.Image

    @classmethod
    def from_context(
        cls,
        context: DetailPreviewContext,
        max_preview_resolution: int,
    ) -> DetailPreviewCompositor:
        """Create a compositor with detailer background work precomputed."""

        source_image = _image_tensor_to_rgb_pil(context.image)
        sampled_region = context.sampled_region or context.work_region
        geometry = build_detail_preview_geometry(
            source_width=source_image.width,
            source_height=source_image.height,
            crop_region=sampled_region,
            max_preview_resolution=max_preview_resolution,
            outline_region=context.work_region,
        )
        preview_image = source_image.resize(
            geometry.preview_size,
            Image.Resampling.BILINEAR,
        )
        wash = Image.new("RGB", preview_image.size, "black")
        washed_background = Image.blend(
            preview_image,
            wash,
            DETAIL_PREVIEW_WASH_OPACITY,
        )
        left, top, right, bottom = geometry.crop_box
        crop_size = (max(1, right - left), max(1, bottom - top))
        detail_alpha_mask = _detail_alpha_mask(
            context.work_mask,
            source_size=(source_image.width, source_image.height),
            preview_size=geometry.preview_size,
            sampled_box=geometry.crop_box,
            target_size=crop_size,
        )
        return cls(
            geometry=geometry,
            washed_background=washed_background,
            detail_alpha_mask=detail_alpha_mask,
        )

    def compose(self, crop_preview: Image.Image) -> Image.Image:
        """Return one full-context preview with the active detail pasted in."""

        left, top, right, bottom = self.geometry.crop_box
        crop_width = max(1, right - left)
        crop_height = max(1, bottom - top)
        resized_crop = crop_preview.convert("RGB").resize(
            (crop_width, crop_height),
            Image.Resampling.BILINEAR,
        )
        output = self.washed_background.copy()
        output.paste(resized_crop, (left, top), self.detail_alpha_mask)
        ImageDraw.Draw(output).rectangle(
            self.geometry.outline_box,
            outline=DETAIL_PREVIEW_OUTLINE_RGB,
            width=DETAIL_PREVIEW_OUTLINE_WIDTH,
        )
        return output


def fit_preview_size(width: int, height: int, max_size: int) -> tuple[int, int]:
    """Fit dimensions inside the preview limit while preserving aspect ratio."""

    _validate_positive_int("width", width)
    _validate_positive_int("height", height)
    _validate_positive_int("max_size", max_size)

    long_side = max(width, height)
    if long_side <= max_size:
        return width, height

    scale = float(max_size) / float(long_side)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def build_detail_preview_geometry(
    source_width: int,
    source_height: int,
    crop_region: CropRegion,
    max_preview_resolution: int,
    outline_region: CropRegion | None = None,
) -> DetailPreviewGeometry:
    """Build preview-space boxes for the active detail work."""

    _validate_positive_int("source_width", source_width)
    _validate_positive_int("source_height", source_height)
    preview_size = fit_preview_size(
        source_width,
        source_height,
        max_preview_resolution,
    )
    _validate_crop_region(crop_region, source_width, source_height)
    resolved_outline_region = outline_region or crop_region
    _validate_crop_region(resolved_outline_region, source_width, source_height)

    preview_width, preview_height = preview_size
    scale_x = float(preview_width) / float(source_width)
    scale_y = float(preview_height) / float(source_height)
    crop_box = _map_crop_box(crop_region, scale_x, scale_y, preview_size)
    outline_box = _map_outline_box(
        resolved_outline_region,
        source_width,
        source_height,
        scale_x,
        scale_y,
        preview_size,
    )
    return DetailPreviewGeometry(
        source_size=(source_width, source_height),
        preview_size=preview_size,
        crop_box=crop_box,
        outline_box=outline_box,
    )


def work_region_from_mask(mask: torch.Tensor) -> CropRegion:
    """Return the tight work region around a non-empty HW or single-item BHW mask."""

    working = _normalize_mask_tensor(mask)
    coordinates = torch.nonzero(working > 0, as_tuple=False)
    if coordinates.numel() == 0:
        raise ValueError("detail preview work mask must contain at least one pixel.")
    top = int(coordinates[:, 0].min().item())
    bottom = int(coordinates[:, 0].max().item()) + 1
    left = int(coordinates[:, 1].min().item())
    right = int(coordinates[:, 1].max().item()) + 1
    return CropRegion(left, top, right, bottom)


def prepare_detail_preview_callback(
    model: Any,
    steps: int,
    context: DetailPreviewContext,
) -> DetailPreviewCallback:
    """Create a ComfyUI sampler callback with full-image detail previews."""

    latent_preview = _latent_preview()
    comfy_utils = _comfy_utils()
    progress = comfy_utils.ProgressBar(steps)
    previewer = latent_preview.get_previewer(
        _model_load_device(model),
        _model_latent_format(model),
    )

    if previewer is None:
        return _progress_only_callback(progress)

    compositor = DetailPreviewCompositor.from_context(
        context,
        max_preview_resolution=latent_preview.MAX_PREVIEW_RESOLUTION,
    )

    def callback(
        step: int,
        x0: torch.Tensor,
        x: torch.Tensor,
        total_steps: int,
    ) -> None:
        """Send one composed full-context preview to ComfyUI."""

        del x
        crop_preview = previewer.decode_latent_to_preview(x0)
        full_preview = compositor.compose(crop_preview)
        progress.update_absolute(
            step + 1,
            total_steps,
            ("JPEG", full_preview, latent_preview.MAX_PREVIEW_RESOLUTION),
        )

    return callback


def _progress_only_callback(progress: ProgressBar) -> DetailPreviewCallback:
    """Create a callback that reports progress without preview images."""

    def callback(
        step: int,
        x0: torch.Tensor,
        x: torch.Tensor,
        total_steps: int,
    ) -> None:
        """Update progress when ComfyUI has no active previewer."""

        del x0, x
        progress.update_absolute(step + 1, total_steps, None)

    return callback


def _map_crop_box(
    crop_region: CropRegion,
    scale_x: float,
    scale_y: float,
    preview_size: tuple[int, int],
) -> CropBox:
    """Map a source crop to a PIL paste box with exclusive right/bottom."""

    preview_width, preview_height = preview_size
    left = _clamp(round(crop_region.left * scale_x), 0, preview_width - 1)
    top = _clamp(round(crop_region.top * scale_y), 0, preview_height - 1)
    right = _clamp(round(crop_region.right * scale_x), left + 1, preview_width)
    bottom = _clamp(round(crop_region.bottom * scale_y), top + 1, preview_height)
    return left, top, right, bottom


def _map_outline_box(
    crop_region: CropRegion,
    source_width: int,
    source_height: int,
    scale_x: float,
    scale_y: float,
    preview_size: tuple[int, int],
) -> CropBox:
    """Map the preferred outside crop outline to preview coordinates."""

    preview_width, preview_height = preview_size
    source_outline = _source_outline_box(crop_region, source_width, source_height)
    left = _clamp(round(source_outline[0] * scale_x), 0, preview_width - 1)
    top = _clamp(round(source_outline[1] * scale_y), 0, preview_height - 1)
    right = _clamp(round(source_outline[2] * scale_x), 0, preview_width - 1)
    bottom = _clamp(round(source_outline[3] * scale_y), 0, preview_height - 1)
    if right < left:
        right = left
    if bottom < top:
        bottom = top
    return left, top, right, bottom


def _source_outline_box(
    crop_region: CropRegion,
    source_width: int,
    source_height: int,
) -> CropBox:
    """Return source-space outline coordinates, preferring outside placement."""

    return (
        crop_region.left - 1 if crop_region.left > 0 else crop_region.left,
        crop_region.top - 1 if crop_region.top > 0 else crop_region.top,
        crop_region.right
        if crop_region.right < source_width
        else crop_region.right - 1,
        crop_region.bottom
        if crop_region.bottom < source_height
        else crop_region.bottom - 1,
    )


def _image_tensor_to_rgb_pil(image: torch.Tensor) -> Image.Image:
    """Convert a single-image BHWC tensor to an RGB PIL image."""

    import numpy as np

    if image.ndim != 4:
        raise ValueError("detail preview image must be a BHWC tensor.")
    if int(image.shape[0]) != 1:
        raise ValueError("detail preview image must contain exactly one image.")
    if int(image.shape[-1]) < 1:
        raise ValueError("detail preview image must contain at least one channel.")

    array = image[0].detach().cpu().float().clamp(0.0, 1.0).numpy()
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] >= 3:
        array = array[..., :3]
    else:
        array = np.repeat(array[..., :1], 3, axis=-1)
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


def _mask_tensor_to_l_pil(mask: torch.Tensor) -> Image.Image:
    """Convert an HW or single-item BHW mask tensor to a grayscale alpha image."""

    import numpy as np

    working = _normalize_mask_tensor(mask).detach().cpu().clamp(0.0, 1.0)
    return Image.fromarray((working.numpy() * 255.0).round().astype(np.uint8))


def _normalize_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    """Normalize an HW or single-item BHW mask tensor to HW float."""

    working = mask.detach().float()
    if working.ndim == 3 and int(working.shape[0]) == 1:
        working = working[0]
    if working.ndim != 2:
        raise ValueError("detail preview work mask must be an HW tensor.")
    return working


def _detail_alpha_mask(
    mask: torch.Tensor,
    *,
    source_size: tuple[int, int],
    preview_size: tuple[int, int],
    sampled_box: CropBox,
    target_size: tuple[int, int],
) -> Image.Image:
    """Return an alpha mask aligned to the sampled preview paste box."""

    mask_image = _mask_tensor_to_l_pil(mask)
    if mask_image.size == source_size:
        preview_mask = mask_image.resize(preview_size, Image.Resampling.BILINEAR)
        return preview_mask.crop(sampled_box).resize(
            target_size,
            Image.Resampling.BILINEAR,
        )
    return mask_image.resize(target_size, Image.Resampling.BILINEAR)


def _validate_crop_region(
    crop_region: CropRegion,
    source_width: int,
    source_height: int,
) -> None:
    """Reject crop regions that cannot be mapped into the source image."""

    if crop_region.left < 0 or crop_region.top < 0:
        raise ValueError("crop_region left and top must be non-negative.")
    if crop_region.right <= crop_region.left or crop_region.bottom <= crop_region.top:
        raise ValueError("crop_region right/bottom must be greater than left/top.")
    if crop_region.right > source_width or crop_region.bottom > source_height:
        raise ValueError("crop_region must fit within the source image.")


def _validate_positive_int(name: str, value: int) -> None:
    """Reject non-positive integer values."""

    if int(value) <= 0:
        raise ValueError(f"{name} must be greater than 0.")


def _clamp(value: int, minimum: int, maximum: int) -> int:
    """Clamp an integer inside inclusive bounds."""

    if minimum > maximum:
        return minimum
    return min(max(value, minimum), maximum)


def _model_load_device(model: Any) -> object:
    """Return the ComfyUI model load device for previewer lookup."""

    preview_model = cast(ModelPreviewSource, model)
    return preview_model.load_device


def _model_latent_format(model: Any) -> object:
    """Return the ComfyUI model latent format for previewer lookup."""

    preview_model = cast(ModelPreviewSource, model)
    return preview_model.model.latent_format


def _latent_preview() -> LatentPreviewModule:
    """Import ComfyUI latent preview support lazily."""

    return cast(LatentPreviewModule, import_module("latent_preview"))


def _comfy_utils() -> ComfyUtilsModule:
    """Import ComfyUI utility support lazily."""

    return cast(ComfyUtilsModule, import_module("comfy.utils"))
