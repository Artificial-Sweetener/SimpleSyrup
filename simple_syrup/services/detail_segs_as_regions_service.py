# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file incorporate behavior derived from
# multidiffusion-upscaler-for-automatic1111. See third_party/manifest.toml and
# third_party/NOTICE.md.

"""SEGS detailer orchestration service using regional MultiDiffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ..domain.conditioning_batch import ConditioningBatch
from ..domain.regional_detailing import (
    LatentRegion,
    pair_segments_with_conditioning,
)
from ..domain.segs import CropRegion, coerce_segs
from ..masking.regional_detailing_masks import (
    build_image_regions,
    build_latent_regions,
    feather_image_mask,
    scale_image_regions,
    union_masks,
)
from ..masking.segs_mask_ops import validate_single_image
from ..runtime import regional_multidiffusion_sampling
from ..runtime.detail_previews import DetailPreviewContext, work_region_from_mask
from ..runtime.detail_resize import DetailImageResizer
from ..runtime.detail_sampling import DetailSampler, Latent
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)
OPERATION = "Detail SEGS as Regions"


class RegionalDetailSamplingBoundary(Protocol):
    """Runtime boundary used by Detail SEGS as Regions."""

    def encode(self, vae: Any, pixels: torch.Tensor, tiled: bool) -> Latent:
        """Encode pixels into a latent dictionary."""

    def decode(self, vae: Any, latent: Latent, tiled: bool) -> torch.Tensor:
        """Decode latent samples into pixels."""

    def sample_regions(
        self,
        *,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: Latent,
        regions: tuple[LatentRegion, ...],
        denoise: float,
        global_prompt_weight: float,
        preview_context: DetailPreviewContext | None = None,
        differential_diffusion: bool = False,
    ) -> Latent:
        """Sample one full latent with paired regional conditioning."""


class RegionalDetailResizeBoundary(Protocol):
    """Image resize boundary used by regional detailing."""

    def resize_up(
        self,
        image: torch.Tensor,
        height: int,
        width: int,
        method: str,
    ) -> torch.Tensor:
        """Resize an image using the selected upscale method."""

    def resize_down_lanczos(
        self,
        image: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Resize an image using fixed Lanczos downscaling."""


@dataclass(frozen=True)
class DetailSEGSAsRegionsResult:
    """Return the detailed image from a regional detail pass."""

    image: torch.Tensor


class RegionalDetailSampler:
    """Adapt shared detail sampling helpers to regional MultiDiffusion."""

    def __init__(self, detail_sampler: DetailSampler | None = None) -> None:
        """Create the runtime adapter with injectable encode/decode helper."""

        self._detail_sampler = detail_sampler or DetailSampler()

    def encode(self, vae: Any, pixels: torch.Tensor, tiled: bool) -> Latent:
        """Encode pixels into a latent dictionary."""

        return self._detail_sampler.encode(vae, pixels, tiled)

    def decode(self, vae: Any, latent: Latent, tiled: bool) -> torch.Tensor:
        """Decode latent samples into pixels."""

        return self._detail_sampler.decode(vae, latent, tiled)

    def sample_regions(
        self,
        *,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: Latent,
        regions: tuple[LatentRegion, ...],
        denoise: float,
        global_prompt_weight: float,
        preview_context: DetailPreviewContext | None = None,
        differential_diffusion: bool = False,
    ) -> Latent:
        """Sample one full latent with regional MultiDiffusion."""

        return regional_multidiffusion_sampling.sample_regional_multidiffusion(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            regions=regions,
            denoise=denoise,
            global_prompt_weight=global_prompt_weight,
            preview_context=preview_context,
            differential_diffusion=differential_diffusion,
        )


class DetailSEGSAsRegionsService:
    """Detail provided SEGS through one regional MultiDiffusion pass."""

    def __init__(
        self,
        sampler: RegionalDetailSamplingBoundary | None = None,
        image_resizer: RegionalDetailResizeBoundary | None = None,
    ) -> None:
        """Create the service with injectable collaborators for tests."""

        self._sampler = sampler or RegionalDetailSampler()
        self._image_resizer = image_resizer or DetailImageResizer()

    def detail(
        self,
        image: object,
        segs: object,
        model: Any,
        vae: Any,
        positive: Any,
        negative: Any,
        region_positive: object,
        scale_factor: float,
        upscale_method: str,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        feather: int,
        noise_mask: bool,
        noise_mask_feather: int,
        tiled_encode: bool,
        tiled_decode: bool,
        global_prompt_weight: float,
    ) -> DetailSEGSAsRegionsResult:
        """Run regional MultiDiffusion detailing for provided SEGS."""

        self._validate_sampling_inputs(
            steps=steps,
            denoise=denoise,
            feather=feather,
            noise_mask_feather=noise_mask_feather,
            global_prompt_weight=global_prompt_weight,
            scale_factor=scale_factor,
        )
        image_tensor = validate_single_image(image, OPERATION)
        native_segs = coerce_segs(segs)
        image_height = int(image_tensor.shape[1])
        image_width = int(image_tensor.shape[2])
        scaled_height, scaled_width = _scaled_dimensions(
            image_height,
            image_width,
            scale_factor,
        )
        scaling_active = (scaled_height, scaled_width) != (image_height, image_width)
        pairs = pair_segments_with_conditioning(
            native_segs,
            region_positive,
            image_height=image_height,
            image_width=image_width,
        )
        if not pairs:
            return DetailSEGSAsRegionsResult(image=image_tensor.clone())
        if not isinstance(region_positive, ConditioningBatch):
            raise TypeError(
                f"{OPERATION} requires region_positive to be CONDITIONING_BATCH."
            )

        image_regions = build_image_regions(
            pairs,
            image_height=image_height,
            image_width=image_width,
        )
        image_union_mask = union_masks(
            tuple(region.image_mask for region in image_regions)
        )
        working_image = (
            self._image_resizer.resize_up(
                image_tensor,
                scaled_height,
                scaled_width,
                upscale_method,
            )
            if scaling_active
            else image_tensor
        )
        sampling_regions = (
            scale_image_regions(
                image_regions,
                image_height=scaled_height,
                image_width=scaled_width,
            )
            if scaling_active
            else image_regions
        )
        latent = self._sampler.encode(vae, working_image, tiled_encode)
        samples = latent.get("samples")
        if not isinstance(samples, torch.Tensor):
            raise ValueError("latent samples must be a torch tensor.")
        latent_height = int(samples.shape[-2])
        latent_width = int(samples.shape[-1])
        latent_regions = build_latent_regions(
            sampling_regions,
            latent_height=latent_height,
            latent_width=latent_width,
            device=samples.device,
            dtype=samples.dtype,
            latent_feather=noise_mask_feather,
        )

        latent_for_sampling = (
            self._with_noise_mask(latent, latent_regions) if noise_mask else latent
        )
        differential_diffusion = noise_mask and noise_mask_feather > 0

        sampled = self._sampler.sample_regions(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_for_sampling,
            regions=latent_regions,
            denoise=denoise,
            global_prompt_weight=global_prompt_weight,
            preview_context=DetailPreviewContext(
                image=image_tensor,
                work_region=work_region_from_mask(image_union_mask),
                work_mask=image_union_mask,
                sampled_region=CropRegion(0, 0, image_width, image_height),
            ),
            differential_diffusion=differential_diffusion,
        )
        decoded = self._sampler.decode(vae, sampled, tiled_decode)
        if decoded.shape[1:3] != image_tensor.shape[1:3]:
            decoded = self._image_resizer.resize_down_lanczos(
                decoded,
                image_height,
                image_width,
            )
        detailed = self._composite_full_image(
            image_tensor,
            decoded,
            feather_image_mask(image_union_mask, feather),
        )

        LOGGER.info(
            "Detail SEGS as Regions pass completed",
            extra={
                "operation": "detail_segs_as_regions",
                "segment_count": len(pairs),
                "latent_width": latent_width,
                "latent_height": latent_height,
                "latent_ndim": samples.ndim,
                "sampler": sampler_name,
                "scheduler": scheduler,
                "steps": steps,
                "denoise": denoise,
                "noise_mask": noise_mask,
                "region_positive_count": len(region_positive.entries),
                "global_prompt_weight": global_prompt_weight,
                "scale_factor": scale_factor,
                "upscale_method": upscale_method,
                "original_width": image_width,
                "original_height": image_height,
                "scaled_width": scaled_width,
                "scaled_height": scaled_height,
            },
        )
        return DetailSEGSAsRegionsResult(image=detailed)

    def _with_noise_mask(
        self,
        latent: Latent,
        regions: tuple[LatentRegion, ...],
    ) -> Latent:
        """Attach a latent-sized union denoise mask to a latent dictionary."""

        samples = latent.get("samples")
        if not isinstance(samples, torch.Tensor):
            raise ValueError("latent samples must be a torch tensor.")
        latent_mask = union_masks(tuple(region.latent_mask for region in regions))
        output = latent.copy()
        output["noise_mask"] = latent_mask.unsqueeze(0).to(
            device=samples.device,
            dtype=samples.dtype,
        )
        return output

    def _composite_full_image(
        self,
        original: torch.Tensor,
        decoded: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Composite a full decoded image through a full-image HW mask."""

        if decoded.ndim != 4:
            raise ValueError("decoded image must be a BHWC tensor.")
        working = decoded
        if working.shape[1:3] != original.shape[1:3]:
            working = self._image_resizer.resize_down_lanczos(
                working,
                int(original.shape[1]),
                int(original.shape[2]),
            )
        alpha = mask.to(device=original.device, dtype=original.dtype).unsqueeze(0)
        alpha = alpha.unsqueeze(-1).clamp(0.0, 1.0)
        return (
            (working.to(device=original.device, dtype=original.dtype) * alpha)
            .add(original * (1.0 - alpha))
            .clamp(0.0, 1.0)
        )

    def _validate_sampling_inputs(
        self,
        *,
        steps: int,
        denoise: float,
        feather: int,
        noise_mask_feather: int,
        global_prompt_weight: float,
        scale_factor: float,
    ) -> None:
        """Reject invalid sampling controls before side effects."""

        if steps < 1:
            raise ValueError("steps must be at least 1.")
        if not 0.0 <= denoise <= 1.0:
            raise ValueError("denoise must be between 0 and 1.")
        if feather < 0:
            raise ValueError("feather must be greater than or equal to 0.")
        if noise_mask_feather < 0:
            raise ValueError("noise_mask_feather must be greater than or equal to 0.")
        if not 0.0 <= global_prompt_weight <= 1.0:
            raise ValueError("global_prompt_weight must be between 0.0 and 1.0.")
        if scale_factor <= 0.0:
            raise ValueError("scale_factor must be greater than 0.")


def _scaled_dimensions(
    image_height: int,
    image_width: int,
    scale_factor: float,
) -> tuple[int, int]:
    """Return detail canvas dimensions for a regional scale factor."""

    if scale_factor <= 1.0:
        return image_height, image_width
    return (
        max(1, int(round(image_height * scale_factor))),
        max(1, int(round(image_width * scale_factor))),
    )
