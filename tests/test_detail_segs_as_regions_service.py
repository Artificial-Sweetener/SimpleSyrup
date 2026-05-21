# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Detail SEGS as Regions service."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.regional_detailing import LatentRegion
from simple_syrup.domain.segs import BoundingBox, CropRegion, NativeSegs, Segment
from simple_syrup.runtime.detail_previews import DetailPreviewContext
from simple_syrup.runtime.detail_sampling import Latent
from simple_syrup.services.detail_segs_as_regions_service import (
    DetailSEGSAsRegionsService,
    RegionalDetailResizeBoundary,
    RegionalDetailSamplingBoundary,
)


def test_empty_segs_return_original_image_without_sampling() -> None:
    """No SEGS regions leave the input image unchanged."""

    sampler = _FakeRegionalSampler()
    service = _service(sampler)
    image = _image()

    result = service.detail(
        image,
        ((8, 8), ()),
        object(),
        object(),
        [],
        [],
        object(),
        **_settings(),
    )

    assert torch.equal(result.image, image)
    assert sampler.encoded_shapes == []
    assert sampler.sample_calls == []


def test_mismatched_region_conditioning_count_fails_before_encode() -> None:
    """Regional prompts must pair exactly with SEGS."""

    sampler = _FakeRegionalSampler()
    service = _service(sampler)

    with pytest.raises(ValueError, match="1 conditioning entries for 2 SEGS"):
        service.detail(
            _image(),
            _segs(_segment(), _segment(CropRegion(4, 4, 8, 8))),
            object(),
            object(),
            [],
            [],
            ConditioningBatch(("only one",)),
            **_settings(),
        )

    assert sampler.encoded_shapes == []


def test_full_image_encoded_once_and_decoded_once() -> None:
    """Regional detailing samples one full latent instead of per-crop latents."""

    sampler = _FakeRegionalSampler()
    service = _service(sampler)

    result = service.detail(
        _image(),
        _segs(_segment()),
        "model",
        "vae",
        "global positive",
        "negative",
        ConditioningBatch(("regional positive",)),
        **_settings(),
    )

    assert result.image.shape == (1, 8, 8, 3)
    assert sampler.encoded_shapes == [(1, 8, 8, 3)]
    assert sampler.decode_count == 1
    assert len(sampler.sample_calls) == 1


def test_regions_are_passed_to_runtime_in_original_order() -> None:
    """SEGS and conditioning remain paired through service orchestration."""

    sampler = _FakeRegionalSampler()
    service = _service(sampler)
    first = _segment(CropRegion(0, 0, 4, 4), label="first")
    second = _segment(CropRegion(4, 4, 8, 8), label="second")

    service.detail(
        _image(),
        _segs(first, second),
        "model",
        "vae",
        "global positive",
        "negative",
        ConditioningBatch(("positive 1", "positive 2")),
        **_settings(),
    )

    regions = sampler.sample_calls[0]["regions"]
    assert [region.index for region in regions] == [0, 1]
    assert [region.label for region in regions] == ["first", "second"]
    assert [region.positive for region in regions] == ["positive 1", "positive 2"]


def test_decoded_output_is_composited_only_inside_union_mask() -> None:
    """Final compositing preserves original pixels outside SEGS masks."""

    sampler = _FakeRegionalSampler()
    service = _service(sampler)

    result = service.detail(
        _image(),
        _segs(_segment(CropRegion(0, 0, 4, 4))),
        "model",
        "vae",
        "global positive",
        "negative",
        ConditioningBatch(("regional positive",)),
        **(_settings() | {"feather": 0}),
    )

    assert torch.all(result.image[:, :4, :4, :] == 1.0)
    assert torch.all(result.image[:, 4:, :, :] == 0.0)
    assert torch.all(result.image[:, :, 4:, :] == 0.0)


def test_scale_factor_resizes_working_canvas_and_preserves_outside_mask() -> None:
    """Scaled regional detailing keeps original pixels outside the union mask."""

    sampler = _FakeRegionalSampler()
    image_resizer = _FakeImageResizer()
    service = _service(sampler, image_resizer)
    image = _image() + 0.25

    result = service.detail(
        image,
        _segs(_segment(CropRegion(0, 0, 4, 4))),
        "model",
        "vae",
        "global positive",
        "negative",
        ConditioningBatch(("regional positive",)),
        **(_settings() | {"scale_factor": 2.0, "upscale_method": "bicubic"}),
    )

    assert sampler.encoded_shapes == [(1, 16, 16, 3)]
    assert image_resizer.upscale_calls == [((1, 8, 8, 3), 16, 16, "bicubic")]
    assert image_resizer.downscale_calls == [((1, 16, 16, 3), 8, 8)]
    assert torch.all(result.image[:, :4, :4, :] == 1.0)
    assert torch.all(result.image[:, 4:, :, :] == 0.25)
    assert torch.all(result.image[:, :, 4:, :] == 0.25)


def test_regional_detailing_forwards_union_preview_context() -> None:
    """Regional sampling previews use original-resolution union-mask context."""

    sampler = _FakeRegionalSampler()
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(_segment(CropRegion(2, 1, 6, 5))),
        "model",
        "vae",
        "global positive",
        "negative",
        ConditioningBatch(("regional positive",)),
        **_settings(),
    )

    preview_context = sampler.sample_calls[0]["preview_context"]
    assert isinstance(preview_context, DetailPreviewContext)
    assert preview_context.work_region == CropRegion(2, 1, 6, 5)
    assert preview_context.sampled_region == CropRegion(0, 0, 8, 8)
    assert preview_context.work_mask.shape == (8, 8)


def test_scale_factor_at_or_below_one_samples_original_size() -> None:
    """Regional scale factors below one do not downscale the working image."""

    sampler = _FakeRegionalSampler()
    image_resizer = _FakeImageResizer()
    service = _service(sampler, image_resizer)

    service.detail(
        _image(),
        _segs(_segment()),
        "model",
        "vae",
        "global positive",
        "negative",
        ConditioningBatch(("regional positive",)),
        **(_settings() | {"scale_factor": 0.5}),
    )

    assert sampler.encoded_shapes == [(1, 8, 8, 3)]
    assert image_resizer.upscale_calls == []
    assert image_resizer.downscale_calls == []


def test_invalid_scale_factor_fails_before_encode() -> None:
    """Scale factor must be positive before runtime side effects."""

    sampler = _FakeRegionalSampler()
    service = _service(sampler)

    with pytest.raises(ValueError, match="scale_factor"):
        service.detail(
            _image(),
            _segs(_segment()),
            "model",
            "vae",
            "global positive",
            "negative",
            ConditioningBatch(("regional positive",)),
            **(_settings() | {"scale_factor": 0.0}),
        )

    assert sampler.encoded_shapes == []


def test_noise_mask_true_attaches_latent_union_mask() -> None:
    """Noise-mask mode passes a latent mask to regional sampling."""

    sampler = _FakeRegionalSampler()
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(_segment()),
        "model",
        "vae",
        "global positive",
        "negative",
        ConditioningBatch(("regional positive",)),
        **_settings(),
    )

    latent_image = sampler.sample_calls[0]["latent_image"]
    assert latent_image["noise_mask"].shape == (1, 4, 4)


def test_noise_mask_false_omits_latent_mask_but_composites_pixels() -> None:
    """Disabling denoise masks still preserves pixels outside region composite."""

    sampler = _FakeRegionalSampler()
    service = _service(sampler)

    result = service.detail(
        _image(),
        _segs(_segment(CropRegion(0, 0, 4, 4))),
        "model",
        "vae",
        "global positive",
        "negative",
        ConditioningBatch(("regional positive",)),
        **(_settings() | {"noise_mask": False, "feather": 0}),
    )

    assert "noise_mask" not in sampler.sample_calls[0]["latent_image"]
    assert torch.all(result.image[:, :4, :4, :] == 1.0)
    assert torch.all(result.image[:, 4:, :, :] == 0.0)


def test_noise_mask_feather_applies_differential_diffusion() -> None:
    """Feathered denoise masks request differential diffusion patching."""

    sampler = _FakeRegionalSampler()
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(_segment()),
        "model",
        "vae",
        "global positive",
        "negative",
        ConditioningBatch(("regional positive",)),
        **(_settings() | {"noise_mask": True, "noise_mask_feather": 2}),
    )

    assert sampler.patch_count == 1
    assert sampler.sample_calls[0]["model"] == "patched model"


def test_encode_decode_and_sampler_controls_are_forwarded() -> None:
    """Service forwards controls to runtime boundary unchanged."""

    sampler = _FakeRegionalSampler()
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(_segment()),
        "model",
        "vae",
        "global positive",
        "negative",
        ConditioningBatch(("regional positive",)),
        **(
            _settings()
            | {
                "tiled_encode": True,
                "tiled_decode": True,
                "seed": 999,
                "steps": 12,
                "cfg": 6.5,
                "sampler_name": "dpmpp_2m",
                "scheduler": "sgm_uniform",
                "denoise": 0.35,
            }
        ),
    )

    assert sampler.encode_tiled == [True]
    assert sampler.decode_tiled == [True]
    call = sampler.sample_calls[0]
    assert call["seed"] == 999
    assert call["steps"] == 12
    assert call["cfg"] == 6.5
    assert call["sampler_name"] == "dpmpp_2m"
    assert call["scheduler"] == "sgm_uniform"
    assert call["denoise"] == 0.35
    assert call["global_prompt_weight"] == 0.25


def test_global_prompt_weight_is_forwarded_to_runtime() -> None:
    """Service forwards the global/regional blend weight unchanged."""

    sampler = _FakeRegionalSampler()
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(_segment()),
        "model",
        "vae",
        "global positive",
        "negative",
        ConditioningBatch(("regional positive",)),
        **(_settings() | {"global_prompt_weight": 0.6}),
    )

    assert sampler.sample_calls[0]["global_prompt_weight"] == 0.6


def test_global_prompt_weight_out_of_range_fails_before_encode() -> None:
    """Global prompt weight must stay in the normalized blend range."""

    sampler = _FakeRegionalSampler()
    service = _service(sampler)

    with pytest.raises(ValueError, match="global_prompt_weight"):
        service.detail(
            _image(),
            _segs(_segment()),
            "model",
            "vae",
            "global positive",
            "negative",
            ConditioningBatch(("regional positive",)),
            **(_settings() | {"global_prompt_weight": 1.25}),
        )

    assert sampler.encoded_shapes == []


def test_image_batch_fails_clearly() -> None:
    """The service handles one image at a time."""

    service = _service(_FakeRegionalSampler())

    with pytest.raises(ValueError, match="supports one image at a time"):
        service.detail(
            torch.zeros((2, 8, 8, 3)),
            _segs(_segment()),
            "model",
            "vae",
            "global positive",
            "negative",
            ConditioningBatch(("regional positive",)),
            **_settings(),
        )


class _FakeRegionalSampler:
    """Fake VAE/sampling adapter for regional detailer tests."""

    def __init__(self) -> None:
        """Create empty call records."""

        self.encoded_shapes: list[tuple[int, ...]] = []
        self.encode_tiled: list[bool] = []
        self.decode_tiled: list[bool] = []
        self.sample_calls: list[dict[str, Any]] = []
        self.decode_count = 0
        self.patch_count = 0

    def encode(self, vae: Any, pixels: torch.Tensor, tiled: bool) -> Latent:
        """Record encoded pixel shape and return a latent."""

        del vae
        self.encoded_shapes.append(tuple(int(dim) for dim in pixels.shape))
        self.encode_tiled.append(tiled)
        return {
            "samples": torch.zeros(
                (
                    1,
                    4,
                    max(1, int(pixels.shape[1]) // 2),
                    max(1, int(pixels.shape[2]) // 2),
                ),
                dtype=torch.float32,
            )
        }

    def decode(self, vae: Any, latent: Latent, tiled: bool) -> torch.Tensor:
        """Return a bright full image matching the latent scale."""

        del vae
        self.decode_count += 1
        self.decode_tiled.append(tiled)
        samples = latent["samples"]
        if not isinstance(samples, torch.Tensor):
            raise ValueError("latent samples must be a tensor.")
        return torch.ones(
            (1, int(samples.shape[-2]) * 2, int(samples.shape[-1]) * 2, 3),
            dtype=torch.float32,
        )

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
    ) -> Latent:
        """Record regional sample options and return the latent unchanged."""

        self.sample_calls.append(
            {
                "model": model,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "positive": positive,
                "negative": negative,
                "latent_image": latent_image,
                "regions": regions,
                "denoise": denoise,
                "global_prompt_weight": global_prompt_weight,
                "preview_context": preview_context,
            }
        )
        return latent_image

    def apply_differential_diffusion(self, model: Any) -> Any:
        """Record patching and return a sentinel model."""

        del model
        self.patch_count += 1
        return "patched model"


class _FakeImageResizer:
    """Fake regional image resizer for service tests."""

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
        """Record upscale options and return a shaped working image."""

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
        """Record downscale options and return a shaped detailed image."""

        self.downscale_calls.append(
            (tuple(int(dim) for dim in image.shape), height, width)
        )
        return torch.ones((int(image.shape[0]), height, width, int(image.shape[3])))


def _service(
    sampler: _FakeRegionalSampler,
    image_resizer: _FakeImageResizer | None = None,
) -> DetailSEGSAsRegionsService:
    """Create a service with fake collaborators."""

    return DetailSEGSAsRegionsService(
        sampler=cast(RegionalDetailSamplingBoundary, sampler),
        image_resizer=cast(
            RegionalDetailResizeBoundary,
            image_resizer or _FakeImageResizer(),
        ),
    )


def _segs(*segments: Segment) -> NativeSegs:
    """Return native SEGS for an 8x8 image."""

    return (8, 8), tuple(segments)


def _segment(
    crop_region: CropRegion | None = None,
    bbox: BoundingBox | None = None,
    label: str = "region",
) -> Segment:
    """Return one test segment with a crop-local mask."""

    resolved_region = crop_region or CropRegion(0, 0, 8, 8)
    return Segment(
        cropped_image=None,
        cropped_mask=torch.ones((resolved_region.height, resolved_region.width)),
        confidence=1.0,
        crop_region=resolved_region,
        bbox=bbox
        or BoundingBox(
            resolved_region.left,
            resolved_region.top,
            resolved_region.right,
            resolved_region.bottom,
        ),
        label=label,
    )


def _image() -> torch.Tensor:
    """Return a dark 8x8 image tensor."""

    return torch.zeros((1, 8, 8, 3), dtype=torch.float32)


def _settings() -> dict[str, Any]:
    """Return valid regional detailer settings for service tests."""

    return {
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
        "global_prompt_weight": 0.25,
        "scale_factor": 1.0,
        "upscale_method": "lanczos",
    }
