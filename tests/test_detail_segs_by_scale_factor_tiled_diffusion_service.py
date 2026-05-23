# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the tiled diffusion scale-factor SEGS detailer service."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from simple_syrup.domain.segs import BoundingBox, CropRegion, NativeSegs, Segment
from simple_syrup.runtime.detail_previews import DetailPreviewContext
from simple_syrup.runtime.detail_sampling import Latent
from simple_syrup.services.detail_segs_by_scale_factor_tiled_diffusion_service import (
    DetailSEGSByScaleFactorTiledDiffusionService,
    TiledDetailResizeBoundary,
    TiledDetailSampler,
    TiledDetailSamplingBoundary,
)


def test_empty_segs_return_original_without_sampling() -> None:
    """Empty SEGS avoid encode and tiled sampling."""

    sampler = _FakeTiledSampler()
    service = _service(sampler)
    image = _image()

    result = service.detail(
        image,
        ((8, 8), ()),
        object(),
        object(),
        [],
        [],
        **_settings(),
    )

    assert torch.equal(result.image, image)
    assert sampler.sample_calls == []
    assert sampler.encoded_shapes == []


def test_multidiffusion_mode_routes_tile_controls() -> None:
    """The tiled detailer forwards MultiDiffusion and tile settings."""

    sampler = _FakeTiledSampler()
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(_segment()),
        "model",
        "vae",
        "positive",
        "negative",
        **(
            _settings()
            | {
                "diffusion_mode": "multidiffusion",
                "latent_tile_width": 64,
                "latent_tile_height": 80,
                "latent_tile_overlap": 12,
                "latent_tile_batch_size": 3,
            }
        ),
    )

    call = sampler.sample_calls[0]
    assert call["diffusion_mode"] == "multidiffusion"
    assert call["latent_tile_width"] == 64
    assert call["latent_tile_height"] == 80
    assert call["latent_tile_overlap"] == 12
    assert call["latent_tile_batch_size"] == 3


def test_mixture_mode_routes_to_tiled_sampler() -> None:
    """Mixture of Diffusers is accepted as a tiled crop mode."""

    sampler = _FakeTiledSampler()
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(_segment()),
        "model",
        "vae",
        "positive",
        "negative",
        **(_settings() | {"diffusion_mode": "mixture_of_diffusers"}),
    )

    assert sampler.sample_calls[0]["diffusion_mode"] == "mixture_of_diffusers"


def test_invalid_diffusion_mode_fails_before_encode() -> None:
    """Unsupported tiled modes fail before runtime side effects."""

    sampler = _FakeTiledSampler()
    service = _service(sampler)

    with pytest.raises(ValueError, match="diffusion_mode"):
        service.detail(
            _image(),
            _segs(_segment()),
            "model",
            "vae",
            "positive",
            "negative",
            **(_settings() | {"diffusion_mode": "full_latent"}),
        )

    assert sampler.encoded_shapes == []


def test_upscale_and_downscale_routing_is_preserved() -> None:
    """Selected upscale and fixed Lanczos downscale are both used."""

    sampler = _FakeTiledSampler()
    image_resizer = _FakeImageResizer()
    service = _service(sampler, image_resizer)

    service.detail(
        _image(),
        _segs(_segment()),
        "model",
        "vae",
        "positive",
        "negative",
        **(_settings() | {"upscale_method": "area"}),
    )

    assert image_resizer.upscale_calls == [((1, 4, 4, 3), 8, 8, "area")]
    assert image_resizer.downscale_calls == [((1, 8, 8, 3), 4, 4)]


def test_tiled_detailer_forwards_detail_preview_context() -> None:
    """Tiled crop sampling uses the shared detailer preview context."""

    sampler = _FakeTiledSampler()
    service = _service(sampler)

    service.detail(_image(), _segs(_segment()), "model", "vae", [], [], **_settings())

    preview_context = sampler.sample_calls[0]["preview_context"]
    assert isinstance(preview_context, DetailPreviewContext)
    assert preview_context.work_region == CropRegion(2, 2, 6, 6)
    assert torch.equal(preview_context.work_mask, torch.ones((4, 4)))


def test_noise_mask_true_attaches_latent_mask() -> None:
    """Noise-mask mode attaches the crop-local mask before tiled sampling."""

    sampler = _FakeTiledSampler()
    service = _service(sampler)

    service.detail(_image(), _segs(_segment()), "model", "vae", [], [], **_settings())

    noise_mask = sampler.sample_calls[0]["latent_image"]["noise_mask"]
    assert noise_mask.shape == (1, 4, 4)


def test_tiled_paste_mask_uses_gaussian_feathered_crop_mask() -> None:
    """Final tiled compositing uses a softened crop-local detailer mask."""

    sampler = _FakeTiledSampler()
    service = _service(sampler)
    mask = torch.zeros((4, 4), dtype=torch.float32)
    mask[1:3, 1:3] = 1.0

    result = service.detail(
        _image(),
        _segs(_segment(cropped_mask=mask)),
        "model",
        "vae",
        [],
        [],
        **(_settings() | {"scale_factor": 1.0, "feather": 1}),
    )

    crop = result.image[:, 2:6, 2:6, 0]
    assert 0.0 < float(crop[0, 0, 1]) < 1.0
    assert float(crop[0, 1, 1]) > float(crop[0, 0, 1])
    assert float(crop[0, 3, 3]) < float(crop[0, 1, 1])


def test_tiled_noise_mask_feather_keeps_sampled_crop_geometry() -> None:
    """Tiled denoise masks keep original crop geometry for ComfyUI resizing."""

    sampler = _FakeTiledSampler()
    service = _service(sampler)
    mask = torch.zeros((4, 4), dtype=torch.float32)
    mask[1:3, 1:3] = 1.0

    service.detail(
        _image(),
        _segs(_segment(cropped_mask=mask)),
        "model",
        "vae",
        [],
        [],
        **(_settings() | {"noise_mask_feather": 1}),
    )

    noise_mask = sampler.sample_calls[0]["latent_image"]["noise_mask"]
    assert noise_mask.shape == (1, 4, 4)
    assert 0.0 < float(noise_mask[0, 0, 1]) < 1.0
    assert float(noise_mask[0, 1, 1]) > float(noise_mask[0, 0, 1])


def test_tiled_detail_sampler_delegates_to_shared_sampling_service() -> None:
    """The detailer adapter uses the shared tiled diffusion dispatch service."""

    tiled_sampling_service = _FakeTiledSamplingService()
    latent = {"samples": torch.zeros((1, 4, 4, 4))}
    preview_context = DetailPreviewContext(
        image=_image(),
        work_region=CropRegion(2, 2, 6, 6),
        work_mask=torch.ones((4, 4)),
    )

    result = TiledDetailSampler(
        tiled_sampling_service=tiled_sampling_service
    ).sample_tiled(
        diffusion_mode="mixture_of_diffusers",
        model="model",
        seed=123,
        steps=4,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        positive="positive",
        negative="negative",
        latent_image=latent,
        denoise=0.5,
        latent_tile_width=128,
        latent_tile_height=80,
        latent_tile_overlap=12,
        latent_tile_batch_size=3,
        preview_context=preview_context,
    )

    assert result is latent
    call = tiled_sampling_service.calls[0]
    assert call["diffusion_mode"] == "mixture_of_diffusers"
    assert call["latent_image"] is latent
    assert call["preview_context"] is preview_context


class _FakeTiledSamplingService:
    """Fake shared tiled diffusion sampling service."""

    def __init__(self) -> None:
        """Create empty call records."""

        self.calls: list[dict[str, Any]] = []

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
    ) -> Latent:
        """Record tiled sampling arguments and return the latent unchanged."""

        self.calls.append(
            {
                "diffusion_mode": diffusion_mode,
                "model": model,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "positive": positive,
                "negative": negative,
                "latent_image": latent_image,
                "denoise": denoise,
                "latent_tile_width": latent_tile_width,
                "latent_tile_height": latent_tile_height,
                "latent_tile_overlap": latent_tile_overlap,
                "latent_tile_batch_size": latent_tile_batch_size,
                "preview_context": preview_context,
            }
        )
        return latent_image


class _FakeTiledSampler:
    """Fake tiled sampling boundary for service tests."""

    def __init__(self) -> None:
        """Create empty call records."""

        self.encoded_shapes: list[tuple[int, ...]] = []
        self.sample_calls: list[dict[str, Any]] = []
        self.patch_count = 0

    def encode(self, vae: Any, pixels: torch.Tensor, tiled: bool) -> Latent:
        """Record encoded pixel shape and return a latent."""

        del vae, tiled
        self.encoded_shapes.append(tuple(int(dim) for dim in pixels.shape))
        return {
            "samples": torch.zeros(
                (
                    1,
                    4,
                    max(1, int(pixels.shape[1]) // 2),
                    max(1, int(pixels.shape[2]) // 2),
                )
            ),
            "pixel_shape": tuple(int(dim) for dim in pixels.shape),
        }

    def decode(self, vae: Any, latent: Latent, tiled: bool) -> torch.Tensor:
        """Return a bright crop matching the encoded pixel shape."""

        del vae, tiled
        shape = cast(tuple[int, int, int, int], latent["pixel_shape"])
        return torch.ones(shape, dtype=torch.float32)

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
    ) -> Latent:
        """Record tiled sample options and return the latent unchanged."""

        self.sample_calls.append(
            {
                "diffusion_mode": diffusion_mode,
                "model": model,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "positive": positive,
                "negative": negative,
                "latent_image": latent_image,
                "denoise": denoise,
                "latent_tile_width": latent_tile_width,
                "latent_tile_height": latent_tile_height,
                "latent_tile_overlap": latent_tile_overlap,
                "latent_tile_batch_size": latent_tile_batch_size,
                "preview_context": preview_context,
            }
        )
        return latent_image

    def apply_differential_diffusion(self, model: Any) -> Any:
        """Record patching and return the model unchanged."""

        self.patch_count += 1
        return model


class _FakeImageResizer:
    """Fake detail image resizer for tiled service tests."""

    def __init__(self) -> None:
        """Create empty call records."""

        self.upscale_calls: list[tuple[tuple[int, ...], int, int, str]] = []
        self.downscale_calls: list[tuple[tuple[int, ...], int, int]] = []

    def resize_up(
        self,
        image: torch.Tensor,
        height: int,
        width: int,
        method: str,
    ) -> torch.Tensor:
        """Record upscale options and return a shaped crop."""

        self.upscale_calls.append(
            (tuple(int(dim) for dim in image.shape), height, width, method)
        )
        return torch.zeros((int(image.shape[0]), height, width, int(image.shape[3])))

    def resize_down_lanczos(
        self,
        image: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Record downscale options and return a shaped detail crop."""

        self.downscale_calls.append(
            (tuple(int(dim) for dim in image.shape), height, width)
        )
        return torch.ones((int(image.shape[0]), height, width, int(image.shape[3])))


def _service(
    sampler: _FakeTiledSampler,
    image_resizer: _FakeImageResizer | None = None,
) -> DetailSEGSByScaleFactorTiledDiffusionService:
    """Create a service with fake collaborators."""

    return DetailSEGSByScaleFactorTiledDiffusionService(
        sampler=cast(TiledDetailSamplingBoundary, sampler),
        image_resizer=cast(
            TiledDetailResizeBoundary,
            image_resizer or _FakeImageResizer(),
        ),
    )


def _segs(*segments: Segment) -> NativeSegs:
    """Return native SEGS for an 8x8 image."""

    return (8, 8), tuple(segments)


def _segment(cropped_mask: object | None = None) -> Segment:
    """Return one test segment with a cropped mask."""

    return Segment(
        cropped_image=None,
        cropped_mask=cropped_mask if cropped_mask is not None else torch.ones((4, 4)),
        confidence=1.0,
        crop_region=CropRegion(2, 2, 6, 6),
        bbox=BoundingBox(3, 3, 5, 5),
        label="face",
    )


def _image() -> torch.Tensor:
    """Return a dark 8x8 image tensor."""

    return torch.zeros((1, 8, 8, 3), dtype=torch.float32)


def _settings() -> dict[str, Any]:
    """Return valid tiled detailer settings for service tests."""

    return {
        "scale_factor": 2.0,
        "upscale_method": "lanczos",
        "clamp_size": 0,
        "seed": 123,
        "steps": 4,
        "cfg": 7.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 0.5,
        "feather": 0,
        "noise_mask": True,
        "noise_mask_feather": 0,
        "tiled_encode": False,
        "tiled_decode": False,
        "diffusion_mode": "multidiffusion",
        "latent_tile_width": 128,
        "latent_tile_height": 128,
        "latent_tile_overlap": 16,
        "latent_tile_batch_size": 4,
    }
