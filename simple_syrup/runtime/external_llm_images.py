# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Image adaptation for OpenAI-compatible external LLM vision requests."""

from __future__ import annotations

import base64
from io import BytesIO

import torch
from PIL import Image

from ..shared.tensor_validation import validate_image_tensor


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
