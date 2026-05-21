# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ViTMatte-based mask edge refinement."""

from __future__ import annotations

from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F
from PIL import Image

from .loaded_models import LoadedViTMatteModel, unwrap_vitmatte_model
from .model_device_manager import TorchModelDeviceManager, external_model_inference


class MaskRefinementSettings(Protocol):
    """Settings required by ViTMatte refinement."""

    @property
    def detail_method(self) -> str:
        """Return the selected detail method."""
        ...

    @property
    def detail_erode(self) -> int:
        """Return trimap erosion radius."""
        ...

    @property
    def detail_dilate(self) -> int:
        """Return trimap dilation radius."""
        ...

    @property
    def black_point(self) -> float:
        """Return mask black point."""
        ...

    @property
    def white_point(self) -> float:
        """Return mask white point."""
        ...

    @property
    def process_detail(self) -> bool:
        """Return whether detail processing is enabled."""
        ...

    @property
    def execution_device(self) -> str:
        """Return execution device policy."""
        ...

    @property
    def max_size_pixels(self) -> int:
        """Return maximum refinement pixel count."""
        ...


class MaskDetailRefiner(Protocol):
    """Refine prompt masks with optional external detail models."""

    def refine(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        settings: MaskRefinementSettings,
        vitmatte_model: object | None,
    ) -> torch.Tensor:
        """Return a refined BHW mask."""


class ViTMatteRefiner:
    """Refine mask edges with a connected ViTMatte model."""

    def refine(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        settings: MaskRefinementSettings,
        vitmatte_model: object | None,
    ) -> torch.Tensor:
        """Run ViTMatte over a batch of image/mask samples."""

        if vitmatte_model is None:
            raise ValueError(
                "detail_method 'VITMatte' requires a connected VITMATTE_MODEL when "
                "refine_mask is enabled. Connect ViTMatte Model Loader or choose "
                "GuidedFilter."
            )

        loaded = unwrap_vitmatte_model(vitmatte_model)
        output_device = mask.device
        refined = [
            self._refine_sample(sample, sample_mask, settings, loaded)
            for sample, sample_mask in zip(image, mask, strict=True)
        ]
        return torch.stack(refined).to(device=output_device, dtype=torch.float32)

    def _refine_sample(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        settings: MaskRefinementSettings,
        loaded_model: LoadedViTMatteModel,
    ) -> torch.Tensor:
        """Refine one image and mask sample."""

        height = int(mask.shape[-2])
        width = int(mask.shape[-1])
        working_image = image.detach().cpu().float().clamp(0.0, 1.0)
        working_mask = mask.detach().cpu().float().clamp(0.0, 1.0)
        erode_radius = settings.detail_erode
        dilate_radius = settings.detail_dilate
        max_dimension = max(height, width)
        if max_dimension > settings.max_size_pixels:
            scale = settings.max_size_pixels / float(max_dimension)
            scaled_height = max(1, int(round(height * scale)))
            scaled_width = max(1, int(round(width * scale)))
            working_image = _resize_image_sample(
                working_image, scaled_height, scaled_width
            )
            working_mask = _resize_mask_sample(
                working_mask, scaled_height, scaled_width
            )
            erode_radius = max(1, int(round(settings.detail_erode * scale)))
            dilate_radius = max(1, int(round(settings.detail_dilate * scale)))

        trimap = generate_vitmatte_trimap(
            working_mask,
            erode_radius,
            dilate_radius,
        )
        alpha = _run_vitmatte(
            loaded_model,
            _tensor_to_rgb_pil(working_image),
            _trimap_to_pil(trimap),
            settings.execution_device,
        )
        if tuple(alpha.shape) != (height, width):
            alpha = _resize_mask_sample(alpha, height, width)
        return alpha.float().clamp(0.0, 1.0)


def generate_vitmatte_trimap(
    mask: torch.Tensor,
    erode_radius: int,
    dilate_radius: int,
) -> torch.Tensor:
    """Create a 0/0.5/1 trimap from a rough mask."""

    mask_bchw = mask.float().clamp(0.0, 1.0).reshape(1, 1, *mask.shape[-2:])
    foreground = _erode(mask_bchw, erode_radius).squeeze(0).squeeze(0) > 0.99
    background = _dilate(mask_bchw, dilate_radius).squeeze(0).squeeze(0) < 0.01
    trimap = torch.full_like(mask.float(), 0.5)
    trimap[foreground] = 1.0
    trimap[background] = 0.0
    return trimap


def _run_vitmatte(
    loaded_model: LoadedViTMatteModel,
    image: Image.Image,
    trimap: Image.Image,
    execution_device: str,
) -> torch.Tensor:
    """Run one ViTMatte inference and return an HW alpha tensor."""

    if loaded_model.managed_model is not None:
        manager = TorchModelDeviceManager()
        with manager.inference(loaded_model.managed_model, execution_device) as loaded:
            alpha = _run_vitmatte_on_device(
                loaded.model,
                loaded_model.processor,
                image,
                trimap,
                loaded.device,
            )
    else:
        with external_model_inference(loaded_model.model, execution_device) as loaded:
            alpha = _run_vitmatte_on_device(
                loaded.model,
                loaded_model.processor,
                image,
                trimap,
                loaded.device,
            )

    while alpha.ndim > 2 and 1 in alpha.shape:
        alpha = alpha.squeeze(0)
    if alpha.ndim == 3:
        alpha = alpha[0]
    if alpha.ndim != 2:
        raise ValueError(
            f"ViTMatte returned invalid alpha shape: {tuple(alpha.shape)}."
        )
    return alpha.clamp(0.0, 1.0)


def _run_vitmatte_on_device(
    model: object,
    processor: object,
    image: Image.Image,
    trimap: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """Run callable ViTMatte objects with inputs on `device`."""

    if not callable(processor):
        raise TypeError("VITMATTE_MODEL processor is not callable.")
    inputs = processor(images=image, trimaps=trimap, return_tensors="pt")
    tensor_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in dict(inputs).items()
    }
    with torch.no_grad():
        outputs = cast(Any, model)(**tensor_inputs)
    return torch.as_tensor(outputs.alphas).detach().cpu().float()


def _resize_mask_sample(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize one HW mask sample."""

    return F.interpolate(
        mask.reshape(1, 1, *mask.shape[-2:]),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).reshape(height, width)


def _resize_image_sample(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize one HWC image sample."""

    return (
        F.interpolate(
            image.movedim(-1, 0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .movedim(0, -1)
    )


def _tensor_to_rgb_pil(image: torch.Tensor) -> Image.Image:
    """Convert one HWC image tensor to RGB PIL."""

    import numpy as np

    array = image.detach().cpu().float().clamp(0.0, 1.0).numpy()
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


def _trimap_to_pil(trimap: torch.Tensor) -> Image.Image:
    """Convert an HW trimap tensor to grayscale PIL."""

    import numpy as np

    array = trimap.detach().cpu().float().clamp(0.0, 1.0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


def _dilate(samples: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate BCHW mask samples."""

    if radius <= 0:
        return samples
    padded = F.pad(samples, (radius, radius, radius, radius), value=0.0)
    return F.max_pool2d(padded, kernel_size=radius * 2 + 1, stride=1)


def _erode(samples: torch.Tensor, radius: int) -> torch.Tensor:
    """Erode BCHW mask samples."""

    if radius <= 0:
        return samples
    padded = F.pad(samples, (radius, radius, radius, radius), value=0.0)
    return -F.max_pool2d(-padded, kernel_size=radius * 2 + 1, stride=1)
