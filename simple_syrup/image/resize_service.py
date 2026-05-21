# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application service for resizing ComfyUI image batches."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import torch

from ..runtime.image_resamplers import (
    NativeComfyResampler,
    Processor,
    validate_sampling,
)
from ..runtime.torchlanc_resampler import TorchLanczosResampler
from ..shared.logging import get_logger
from ..shared.tensor_validation import (
    ImageTensorShape,
    validate_image_tensor,
    validate_mask_tensor,
)
from .resize_geometry import (
    CropPosition,
    ResizeMode,
    ResizePlan,
    ResizeTarget,
    build_resize_plan,
)


class ProgressReporter(Protocol):
    """Progress reporting interface used by the resize service."""

    def update(self, value: int) -> None:
        """Advance progress by the given number of items."""


NativeResamplerFactory = Callable[[Processor], NativeComfyResampler]
TorchLancFactory = Callable[[], TorchLanczosResampler]
ProgressFactory = Callable[[int], ProgressReporter]

_LOGGER = get_logger(__name__)


class ResizeImageToTargetService:
    """Resize ComfyUI image batches to a target geometry."""

    def __init__(
        self,
        native_resampler_factory: NativeResamplerFactory | None = None,
        torchlanc_factory: TorchLancFactory | None = None,
        progress_factory: ProgressFactory | None = None,
    ) -> None:
        """Create the service with injectable runtime boundaries."""

        self._native_resampler_factory = native_resampler_factory
        self._torchlanc_factory = torchlanc_factory
        self._progress_factory = progress_factory

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
        """Return resized image, output width, output height, and resized mask."""

        try:
            image_shape = validate_image_tensor(image)
            if mask is not None:
                validate_mask_tensor(mask, image_shape.batch_size)
            validate_sampling(sampling)
            normalized_processor = _coerce_processor(processor)
            normalized_mode = ResizeMode(resize_mode)
            normalized_position = CropPosition(crop_position)
            plan = build_resize_plan(
                image_shape.width,
                image_shape.height,
                ResizeTarget(
                    width=int(width), height=int(height), divisible_by=int(divisible_by)
                ),
                normalized_mode,
                normalized_position,
            )
            _LOGGER.debug(
                "resize_start",
                extra={
                    "source_width": image_shape.width,
                    "source_height": image_shape.height,
                    "output_width": plan.output_width,
                    "output_height": plan.output_height,
                    "mode": normalized_mode.value,
                    "sampling": sampling,
                    "processor": normalized_processor,
                    "batch_size": image_shape.batch_size,
                },
            )
            output = self._resize_validated(
                image=image,
                image_shape=image_shape,
                plan=plan,
                sampling=sampling,
                processor=normalized_processor,
                pad_color=pad_color,
                max_batch_size=int(max_batch_size),
                sinc_window=int(sinc_window),
                precision=precision,
                mask=mask,
            )
            _LOGGER.debug(
                "resize_complete",
                extra={
                    "output_width": plan.output_width,
                    "output_height": plan.output_height,
                    "batch_size": image_shape.batch_size,
                },
            )
            return output
        except Exception:
            _LOGGER.exception(
                "resize_failed",
                extra={
                    "width": width,
                    "height": height,
                    "resize_mode": resize_mode,
                    "sampling": sampling,
                    "processor": processor,
                },
            )
            raise

    def _resize_validated(
        self,
        image: torch.Tensor,
        image_shape: ImageTensorShape,
        plan: ResizePlan,
        sampling: str,
        processor: Processor,
        pad_color: str,
        max_batch_size: int,
        sinc_window: int,
        precision: str,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int, int, torch.Tensor]:
        """Execute resizing after public inputs have been validated."""

        image_bchw = image.float().clamp(0.0, 1.0).movedim(-1, 1)
        mask_bchw = (
            mask.float().clamp(0.0, 1.0).unsqueeze(1) if mask is not None else None
        )
        pad_values = parse_pad_color(pad_color, image_shape.channels)
        output_images: list[torch.Tensor] = []
        output_masks: list[torch.Tensor] = []
        progress = self._create_progress(image_shape.batch_size)

        for start, end in _chunk_spans(image_shape.batch_size, max_batch_size):
            image_chunk = image_bchw[start:end]
            resized = self._resize_image_chunk(
                image_chunk,
                plan,
                sampling,
                processor,
                sinc_window,
                precision,
            )
            finalized = _apply_crop_and_pad_image(resized, plan, pad_values)
            output_images.append(finalized.to("cpu").movedim(1, -1))

            if mask_bchw is not None:
                mask_chunk = mask_bchw[start:end]
                resized_mask = self._native_resampler(processor).resize(
                    mask_chunk,
                    plan.resize_width,
                    plan.resize_height,
                    "nearest-exact",
                )
                finalized_mask = _apply_crop_and_pad_mask(resized_mask, plan)
                output_masks.append(finalized_mask.to("cpu").squeeze(1))

            progress.update(end - start)

        image_out = torch.cat(output_images, dim=0)
        if output_masks:
            mask_out = torch.cat(output_masks, dim=0)
        else:
            mask_out = torch.zeros(
                (image_shape.batch_size, plan.output_height, plan.output_width),
                dtype=torch.float32,
            )

        return image_out, plan.output_width, plan.output_height, mask_out

    def _resize_image_chunk(
        self,
        samples: torch.Tensor,
        plan: ResizePlan,
        sampling: str,
        processor: Processor,
        sinc_window: int,
        precision: str,
    ) -> torch.Tensor:
        """Route one image chunk to the selected runtime backend."""

        with torch.inference_mode():
            if processor == "gpu" and sampling == "lanczos":
                return self._torchlanc_resampler().resize(
                    samples,
                    plan.resize_width,
                    plan.resize_height,
                    sinc_window,
                    precision,
                )

            return self._native_resampler(processor).resize(
                samples,
                plan.resize_width,
                plan.resize_height,
                sampling,
            )

    def _native_resampler(self, processor: Processor) -> NativeComfyResampler:
        """Create the native resampler for the current processor."""

        if self._native_resampler_factory is not None:
            return self._native_resampler_factory(processor)
        return NativeComfyResampler(processor)

    def _torchlanc_resampler(self) -> TorchLanczosResampler:
        """Create the TorchLanc resampler."""

        if self._torchlanc_factory is not None:
            return self._torchlanc_factory()
        return TorchLanczosResampler()

    def _create_progress(self, total: int) -> ProgressReporter:
        """Create a progress reporter for the batch."""

        if self._progress_factory is not None:
            return self._progress_factory(total)

        import comfy.utils

        return cast(ProgressReporter, comfy.utils.ProgressBar(total))


def parse_pad_color(color: str, channels: int) -> torch.Tensor:
    """Parse an RGB padding color into a normalized channel tensor."""

    if channels not in (1, 3, 4):
        raise ValueError(f"Unsupported channel count {channels}. Expected 1, 3, or 4.")

    parts = [part.strip() for part in color.split(",")]
    if len(parts) != 3:
        raise ValueError(
            "pad_color must contain exactly three comma-separated RGB values."
        )

    values: list[float] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(
                f"pad_color value {part!r} is not an integer RGB component."
            ) from exc
        values.append(float(max(0, min(255, value))) / 255.0)

    if channels == 1:
        return torch.tensor([values[0]], dtype=torch.float32)
    if channels == 4:
        return torch.tensor([values[0], values[1], values[2], 1.0], dtype=torch.float32)
    return torch.tensor(values, dtype=torch.float32)


def _apply_crop_and_pad_image(
    samples: torch.Tensor,
    plan: ResizePlan,
    pad_values: torch.Tensor,
) -> torch.Tensor:
    """Apply final crop and pad operations to BCHW image samples."""

    cropped = _crop_samples(samples, plan)
    if not plan.has_pad:
        return cropped

    values = pad_values.to(device=cropped.device, dtype=cropped.dtype).view(1, -1, 1, 1)
    canvas = values.expand(
        cropped.shape[0],
        cropped.shape[1],
        plan.output_height,
        plan.output_width,
    ).clone()
    canvas[
        :,
        :,
        plan.pad_top : plan.pad_top + cropped.shape[-2],
        plan.pad_left : plan.pad_left + cropped.shape[-1],
    ] = cropped
    return canvas


def _apply_crop_and_pad_mask(samples: torch.Tensor, plan: ResizePlan) -> torch.Tensor:
    """Apply final crop and pad operations to BCHW mask samples."""

    cropped = _crop_samples(samples, plan)
    if not plan.has_pad:
        return cropped

    canvas = torch.zeros(
        (cropped.shape[0], 1, plan.output_height, plan.output_width),
        dtype=cropped.dtype,
        device=cropped.device,
    )
    canvas[
        :,
        :,
        plan.pad_top : plan.pad_top + cropped.shape[-2],
        plan.pad_left : plan.pad_left + cropped.shape[-1],
    ] = cropped
    return canvas


def _crop_samples(samples: torch.Tensor, plan: ResizePlan) -> torch.Tensor:
    """Crop BCHW samples according to the resize plan."""

    return samples[
        :,
        :,
        plan.crop_y : plan.crop_y + plan.output_height,
        plan.crop_x : plan.crop_x + plan.output_width,
    ]


def _chunk_spans(batch_size: int, max_batch_size: int) -> list[tuple[int, int]]:
    """Split a batch into inclusive-exclusive chunk spans."""

    if max_batch_size <= 0 or max_batch_size >= batch_size:
        return [(0, batch_size)]
    spans: list[tuple[int, int]] = []
    start = 0
    while start < batch_size:
        end = min(batch_size, start + max_batch_size)
        spans.append((start, end))
        start = end
    return spans


def _coerce_processor(processor: str) -> Processor:
    """Validate and narrow a raw processor string."""

    if processor == "cpu":
        return "cpu"
    if processor == "gpu":
        return "gpu"
    raise ValueError(f"processor must be 'cpu' or 'gpu', got {processor!r}.")
