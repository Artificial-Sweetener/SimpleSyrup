# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SEGS crop detailer orchestration using tiled diffusion sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ..domain.conditioning_batch import select_conditioning
from ..domain.detail_geometry import DetailScalePlan, build_detail_scale_plan
from ..domain.segs import Segment, coerce_segs
from ..domain.tiled_diffusion import validate_tiled_diffusion_mode
from ..image.crop_composite import composite_crop
from ..masking.detailer_masks import gaussian_feather_mask
from ..masking.segs_mask_ops import (
    crop_image,
    validate_single_image,
)
from ..runtime.detail_previews import DetailPreviewContext
from ..runtime.detail_resize import DetailImageResizer
from ..runtime.detail_sampling import DetailSampler, Latent
from ..shared.logging import get_logger
from .tiled_diffusion_sampling_service import TiledDiffusionSamplingService

LOGGER = get_logger(__name__)


class TiledDiffusionLatentSamplingBoundary(Protocol):
    """Latent sampling boundary for selectable tiled diffusion modes."""

    def sample(
        self,
        *,
        diffusion_mode: str,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: Latent,
        denoise: float,
        latent_tile_width: int,
        latent_tile_height: int,
        latent_tile_overlap: int,
        latent_tile_batch_size: int,
        preview_context: DetailPreviewContext | None = None,
        differential_diffusion: bool = False,
    ) -> Latent:
        """Sample a latent using the selected tiled diffusion mode."""


class TiledDetailSamplingBoundary(Protocol):
    """Runtime boundary used by tiled scale-factor detailing."""

    def encode(self, vae: Any, pixels: torch.Tensor, tiled: bool) -> Latent:
        """Encode pixels into a latent dictionary."""

    def decode(self, vae: Any, latent: Latent, tiled: bool) -> torch.Tensor:
        """Decode latent samples into pixels."""

    def sample_tiled(
        self,
        *,
        diffusion_mode: str,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: Latent,
        denoise: float,
        latent_tile_width: int,
        latent_tile_height: int,
        latent_tile_overlap: int,
        latent_tile_batch_size: int,
        preview_context: DetailPreviewContext | None = None,
        differential_diffusion: bool = False,
    ) -> Latent:
        """Sample one latent crop with the requested tiled diffusion mode."""


class TiledDetailResizeBoundary(Protocol):
    """Image resize boundary used by tiled scale-factor detailing."""

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
class TiledDetailerResult:
    """Return the detailed image from a tiled detail pass."""

    image: torch.Tensor


class TiledDetailSampler:
    """Adapt shared VAE helpers and tiled diffusion runtimes."""

    def __init__(
        self,
        detail_sampler: DetailSampler | None = None,
        tiled_sampling_service: TiledDiffusionLatentSamplingBoundary | None = None,
    ) -> None:
        """Create the adapter with injectable encode/decode behavior."""

        self._detail_sampler = detail_sampler or DetailSampler()
        self._tiled_sampling_service = (
            tiled_sampling_service or TiledDiffusionSamplingService()
        )

    def encode(self, vae: Any, pixels: torch.Tensor, tiled: bool) -> Latent:
        """Encode pixels into a latent dictionary."""

        return self._detail_sampler.encode(vae, pixels, tiled)

    def decode(self, vae: Any, latent: Latent, tiled: bool) -> torch.Tensor:
        """Decode latent samples into pixels."""

        return self._detail_sampler.decode(vae, latent, tiled)

    def sample_tiled(
        self,
        *,
        diffusion_mode: str,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: Latent,
        denoise: float,
        latent_tile_width: int,
        latent_tile_height: int,
        latent_tile_overlap: int,
        latent_tile_batch_size: int,
        preview_context: DetailPreviewContext | None = None,
        differential_diffusion: bool = False,
    ) -> Latent:
        """Sample one latent crop with the selected tiled diffusion runtime."""

        return self._tiled_sampling_service.sample(
            diffusion_mode=diffusion_mode,
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
            latent_tile_width=latent_tile_width,
            latent_tile_height=latent_tile_height,
            latent_tile_overlap=latent_tile_overlap,
            latent_tile_batch_size=latent_tile_batch_size,
            preview_context=preview_context,
            differential_diffusion=differential_diffusion,
        )


class DetailSEGSByScaleFactorTiledDiffusionService:
    """Detail SEGS crops with tiled diffusion latent sampling."""

    def __init__(
        self,
        sampler: TiledDetailSamplingBoundary | None = None,
        image_resizer: TiledDetailResizeBoundary | None = None,
    ) -> None:
        """Create the service with injectable collaborators for tests."""

        self._sampler = sampler or TiledDetailSampler()
        self._image_resizer = image_resizer or DetailImageResizer()

    def detail(
        self,
        image: object,
        segs: object,
        model: Any,
        vae: Any,
        positive: Any,
        negative: Any,
        scale_factor: float,
        upscale_method: str,
        clamp_size: int,
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
        diffusion_mode: str,
        latent_tile_width: int,
        latent_tile_height: int,
        latent_tile_overlap: int,
        latent_tile_batch_size: int,
    ) -> TiledDetailerResult:
        """Run crop sampling and composite-back detailing with tiled diffusion."""

        validate_tiled_diffusion_mode(diffusion_mode)
        self._validate_sampling_inputs(
            steps=steps,
            denoise=denoise,
            feather=feather,
            noise_mask_feather=noise_mask_feather,
        )
        image_tensor = validate_single_image(
            image,
            "Detail SEGS by Scale Factor w/ Tiled Diffusion",
        )
        native_segs = coerce_segs(segs)
        _header, segments = native_segs
        if not segments:
            return TiledDetailerResult(image=image_tensor.clone())

        working_image = image_tensor.clone()
        differential_diffusion = noise_mask and noise_mask_feather > 0

        for index, segment in enumerate(segments):
            plan = build_detail_scale_plan(
                detected_width=segment.bbox.width,
                detected_height=segment.bbox.height,
                crop_width=segment.crop_region.width,
                crop_height=segment.crop_region.height,
                scale_factor=scale_factor,
                clamp_size=clamp_size,
            )
            if plan.scale <= 1.0:
                plan = DetailScalePlan(
                    width=segment.crop_region.width,
                    height=segment.crop_region.height,
                    scale=1.0,
                    unclamped_long_side=plan.unclamped_long_side,
                    target_long_side=float(
                        max(segment.crop_region.width, segment.crop_region.height)
                    ),
                )
            working_image = self._detail_segment(
                working_image=working_image,
                segment=segment,
                plan=plan,
                model=model,
                vae=vae,
                positive=select_conditioning(positive, index),
                negative=select_conditioning(negative, index),
                upscale_method=upscale_method,
                seed=seed + index,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                feather=feather,
                noise_mask=noise_mask,
                noise_mask_feather=noise_mask_feather,
                tiled_encode=tiled_encode,
                tiled_decode=tiled_decode,
                diffusion_mode=diffusion_mode,
                latent_tile_width=latent_tile_width,
                latent_tile_height=latent_tile_height,
                latent_tile_overlap=latent_tile_overlap,
                latent_tile_batch_size=latent_tile_batch_size,
                differential_diffusion=differential_diffusion,
            )

        LOGGER.info(
            "Detail SEGS by Scale Factor tiled diffusion pass completed",
            extra={
                "operation": "detail_segs_by_scale_factor_tiled_diffusion",
                "scale_factor": scale_factor,
                "upscale_method": upscale_method,
                "clamp_size": clamp_size,
                "diffusion_mode": diffusion_mode,
                "latent_tile_width": latent_tile_width,
                "latent_tile_height": latent_tile_height,
                "latent_tile_overlap": latent_tile_overlap,
                "latent_tile_batch_size": latent_tile_batch_size,
                "sampler": sampler_name,
                "scheduler": scheduler,
                "segment_count": len(segments),
            },
        )
        return TiledDetailerResult(image=working_image)

    def _detail_segment(
        self,
        working_image: torch.Tensor,
        segment: Segment,
        plan: DetailScalePlan,
        model: Any,
        vae: Any,
        positive: Any,
        negative: Any,
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
        diffusion_mode: str,
        latent_tile_width: int,
        latent_tile_height: int,
        latent_tile_overlap: int,
        latent_tile_batch_size: int,
        differential_diffusion: bool,
    ) -> torch.Tensor:
        """Detail one segment with tiled diffusion and return the updated image."""

        crop = crop_image(working_image, segment.crop_region)
        full_mask = segment.cropped_mask
        if isinstance(full_mask, torch.Tensor):
            mask_crop = full_mask.float()
        else:
            mask_crop = torch.as_tensor(full_mask).float()
        scaled_crop = self._image_resizer.resize_up(
            crop,
            plan.height,
            plan.width,
            upscale_method,
        )
        latent = self._sampler.encode(vae, scaled_crop, tiled_encode)
        if noise_mask:
            latent = self._with_noise_mask(latent, mask_crop, noise_mask_feather)
        sampled = self._sampler.sample_tiled(
            diffusion_mode=diffusion_mode,
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=denoise,
            latent_tile_width=latent_tile_width,
            latent_tile_height=latent_tile_height,
            latent_tile_overlap=latent_tile_overlap,
            latent_tile_batch_size=latent_tile_batch_size,
            differential_diffusion=differential_diffusion,
            preview_context=DetailPreviewContext(
                image=working_image,
                work_region=segment.crop_region,
                work_mask=mask_crop,
            ),
        )
        decoded = self._sampler.decode(vae, sampled, tiled_decode)
        resized_detail = self._image_resizer.resize_down_lanczos(
            decoded,
            segment.crop_region.height,
            segment.crop_region.width,
        )
        paste_mask = gaussian_feather_mask(mask_crop, feather).to(
            device=working_image.device
        )
        return composite_crop(
            image=working_image,
            crop=resized_detail,
            mask=paste_mask,
            region=segment.crop_region,
        )

    def _with_noise_mask(
        self,
        latent: Latent,
        mask: torch.Tensor,
        noise_mask_feather: int,
    ) -> Latent:
        """Attach a crop-space denoise mask to a latent dictionary."""

        samples = latent.get("samples")
        if not isinstance(samples, torch.Tensor):
            raise ValueError("latent samples must be a torch tensor.")
        latent_mask = mask
        if noise_mask_feather > 0:
            latent_mask = gaussian_feather_mask(latent_mask, noise_mask_feather)
        output = latent.copy()
        output["noise_mask"] = latent_mask.unsqueeze(0).to(
            device=samples.device,
            dtype=samples.dtype,
        )
        return output

    def _validate_sampling_inputs(
        self,
        steps: int,
        denoise: float,
        feather: int,
        noise_mask_feather: int,
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
