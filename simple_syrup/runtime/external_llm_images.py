# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Image adaptation for OpenAI-compatible external LLM vision requests."""

from __future__ import annotations

import base64
from io import BytesIO

import torch
from PIL import Image

from ..domain.segs import Segment
from ..masking.segs_mask_ops import crop_image, crop_mask, resize_mask
from ..shared.tensor_validation import validate_image_tensor

SEG_IMAGE_MODES = ("transparent mask", "black mask", "full crop")


class ExternalLLMImageEncoder:
    """Encode ComfyUI IMAGE tensors for OpenAI-compatible vision payloads."""

    def encode_first_image_as_data_url(self, image: object) -> str:
        """Return the first image in a ComfyUI IMAGE batch as a PNG data URL."""

        validate_image_tensor(image)
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor with shape (B, H, W, C).")

        pil_image = _image_tensor_to_rgb_pil(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"


class ExternalLLMSegsImageEncoder:
    """Encode SEG crops for OpenAI-compatible vision payloads."""

    def encode_segment_as_data_url(
        self,
        image: torch.Tensor,
        segment: Segment,
        mode: str,
    ) -> str:
        """Return one SEG crop as a PNG data URL."""

        if mode not in SEG_IMAGE_MODES:
            choices = ", ".join(SEG_IMAGE_MODES)
            raise ValueError(f"seg_image_mode must be one of: {choices}.")
        validate_image_tensor(image)
        if int(image.shape[0]) != 1:
            raise ValueError("SEG crop image encoding requires one IMAGE item.")

        crop = crop_image(image.float().clamp(0.0, 1.0), segment.crop_region)
        if mode == "full crop":
            return _pil_to_png_data_url(_image_tensor_to_rgb_pil(crop))

        mask = _segment_mask_for_crop(
            segment=segment,
            image_height=int(image.shape[1]),
            image_width=int(image.shape[2]),
        )
        if int(mask.shape[0]) != int(crop.shape[1]) or int(mask.shape[1]) != int(
            crop.shape[2]
        ):
            mask = resize_mask(mask, int(crop.shape[1]), int(crop.shape[2]))

        if mode == "black mask":
            masked_crop = crop * mask.unsqueeze(0).unsqueeze(-1)
            return _pil_to_png_data_url(_image_tensor_to_rgb_pil(masked_crop))

        return _pil_to_png_data_url(_crop_and_mask_to_rgba_pil(crop, mask))


def _image_tensor_to_rgb_pil(image: torch.Tensor) -> Image.Image:
    """Convert the first BHWC image tensor item to RGB PIL image."""

    import numpy as np

    array = image[0].detach().cpu().float().clamp(0.0, 1.0).numpy()
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] >= 3:
        array = array[..., :3]
    else:
        array = np.repeat(array[..., :1], 3, axis=-1)
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


def _segment_mask_for_crop(
    segment: Segment,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Return the segment mask normalized to the crop region."""

    if not isinstance(segment.cropped_mask, torch.Tensor):
        raise TypeError("SEG cropped_mask must be a torch.Tensor.")

    mask = _normalize_mask_shape(segment.cropped_mask)
    region = segment.crop_region
    crop_height = region.height
    crop_width = region.width
    mask_height = int(mask.shape[0])
    mask_width = int(mask.shape[1])
    if mask_height == crop_height and mask_width == crop_width:
        return mask.float().clamp(0.0, 1.0)
    if mask_height == image_height and mask_width == image_width:
        return crop_mask(mask, region).float().clamp(0.0, 1.0)
    return resize_mask(mask, crop_height, crop_width).float().clamp(0.0, 1.0)


def _normalize_mask_shape(mask: torch.Tensor) -> torch.Tensor:
    """Return a SEG mask as an HW tensor."""

    working = mask.detach().cpu().float()
    if working.ndim == 2:
        return working
    if working.ndim == 3:
        return working[0]
    raise ValueError("SEG cropped_mask must be an HW or BHW tensor.")


def _crop_and_mask_to_rgba_pil(crop: torch.Tensor, mask: torch.Tensor) -> Image.Image:
    """Convert a BHWC crop and HW alpha mask to an RGBA PIL image."""

    import numpy as np

    rgb = crop[0].detach().cpu().float().clamp(0.0, 1.0).numpy()
    if rgb.shape[-1] == 1:
        rgb = np.repeat(rgb, 3, axis=-1)
    elif rgb.shape[-1] >= 3:
        rgb = rgb[..., :3]
    else:
        rgb = np.repeat(rgb[..., :1], 3, axis=-1)
    alpha = mask.detach().cpu().float().clamp(0.0, 1.0).numpy()
    rgba = np.concatenate((rgb, alpha[..., None]), axis=-1)
    return Image.fromarray((rgba * 255.0).round().astype(np.uint8), mode="RGBA")


def _pil_to_png_data_url(image: Image.Image) -> str:
    """Return a PNG data URL for a PIL image."""

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
