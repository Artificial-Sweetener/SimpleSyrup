# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Detail SEGS by Scale Factor service."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.segs import BoundingBox, CropRegion, NativeSegs, Segment
from simple_syrup.runtime.detail_previews import DetailPreviewContext
from simple_syrup.runtime.detail_sampling import Latent
from simple_syrup.services.detail_segs_by_scale_factor_service import (
    DetailResizeBoundary,
    DetailSamplingBoundary,
    DetailSEGSByScaleFactorService,
)


def test_empty_segs_return_original_image() -> None:
    """No SEGS regions leave the input image unchanged."""

    sampler = _FakeSampler()
    service = _service(sampler)
    image = _image()

    result = service.detail(
        image, ((8, 8), ()), object(), object(), [], [], **_settings()
    )

    assert torch.equal(result.image, image)
    assert sampler.sample_calls == []


def test_one_segment_calls_sampling_once() -> None:
    """One SEGS segment produces one sample call."""

    sampler = _FakeSampler()
    service = _service(sampler)

    result = service.detail(
        _image(), _segs(_segment()), object(), object(), [], [], **_settings()
    )

    assert result.image.shape == (1, 8, 8, 3)
    assert [call.seed for call in sampler.sample_calls] == [123]
    assert sampler.sample_calls[0].preview_context is not None
    assert sampler.sample_calls[0].preview_context.work_region == CropRegion(2, 2, 6, 6)
    assert torch.equal(
        sampler.sample_calls[0].preview_context.work_mask,
        torch.ones((4, 4), dtype=torch.float32),
    )


def test_two_segments_increment_seed_deterministically() -> None:
    """Each processed segment increments the seed by SEGS order."""

    sampler = _FakeSampler()
    first = _segment(CropRegion(0, 0, 4, 4), BoundingBox(1, 1, 3, 3))
    second = _segment(CropRegion(4, 4, 8, 8), BoundingBox(5, 5, 7, 7))
    service = _service(sampler)

    service.detail(
        _image(), _segs(first, second), object(), object(), [], [], **_settings()
    )

    assert [call.seed for call in sampler.sample_calls] == [123, 124]


def test_later_segment_preview_context_uses_current_working_image() -> None:
    """Preview context reflects already-composited earlier segment edits."""

    sampler = _FakeSampler()
    first = _segment(CropRegion(0, 0, 4, 4), BoundingBox(1, 1, 3, 3))
    second = _segment(CropRegion(4, 4, 8, 8), BoundingBox(5, 5, 7, 7))
    service = _service(sampler)

    service.detail(
        _image(), _segs(first, second), object(), object(), [], [], **_settings()
    )

    second_context = sampler.sample_calls[1].preview_context
    assert second_context is not None
    assert torch.all(second_context.image[:, 0:4, 0:4, :] == 1.0)
    assert second_context.work_region == CropRegion(4, 4, 8, 8)


def test_normal_conditioning_broadcasts_to_every_segment() -> None:
    """Normal conditionings are reused for each processed segment."""

    sampler = _FakeSampler()
    first = _segment(CropRegion(0, 0, 4, 4), BoundingBox(1, 1, 3, 3))
    second = _segment(CropRegion(4, 4, 8, 8), BoundingBox(5, 5, 7, 7))
    positive = object()
    negative = object()
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(first, second),
        object(),
        object(),
        positive,
        negative,
        **_settings(),
    )

    assert [call.positive for call in sampler.sample_calls] == [positive, positive]
    assert [call.negative for call in sampler.sample_calls] == [negative, negative]


def test_positive_conditioning_batch_selects_by_segment_index() -> None:
    """A positive batch varies by SEG while normal negative broadcasts."""

    sampler = _FakeSampler()
    first = _segment(CropRegion(0, 0, 4, 4), BoundingBox(1, 1, 3, 3))
    second = _segment(CropRegion(4, 4, 8, 8), BoundingBox(5, 5, 7, 7))
    negative = object()
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(first, second),
        object(),
        object(),
        ConditioningBatch(("positive 1", "positive 2")),
        negative,
        **_settings(),
    )

    assert [call.positive for call in sampler.sample_calls] == [
        "positive 1",
        "positive 2",
    ]
    assert [call.negative for call in sampler.sample_calls] == [negative, negative]


def test_negative_conditioning_batch_selects_by_segment_index() -> None:
    """A negative batch varies by SEG while normal positive broadcasts."""

    sampler = _FakeSampler()
    first = _segment(CropRegion(0, 0, 4, 4), BoundingBox(1, 1, 3, 3))
    second = _segment(CropRegion(4, 4, 8, 8), BoundingBox(5, 5, 7, 7))
    positive = object()
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(first, second),
        object(),
        object(),
        positive,
        ConditioningBatch(("negative 1", "negative 2")),
        **_settings(),
    )

    assert [call.positive for call in sampler.sample_calls] == [positive, positive]
    assert [call.negative for call in sampler.sample_calls] == [
        "negative 1",
        "negative 2",
    ]


def test_conditioning_batches_reuse_last_entry_for_extra_segments() -> None:
    """Batch selection falls back to the last prompt when SEGS outnumber entries."""

    sampler = _FakeSampler()
    first = _segment(CropRegion(0, 0, 4, 4), BoundingBox(1, 1, 3, 3))
    second = _segment(CropRegion(4, 4, 8, 8), BoundingBox(5, 5, 7, 7))
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(first, second),
        object(),
        object(),
        ConditioningBatch(("positive 1",)),
        ConditioningBatch(("negative 1", "negative 2")),
        **_settings(),
    )

    assert [call.positive for call in sampler.sample_calls] == [
        "positive 1",
        "positive 1",
    ]
    assert [call.negative for call in sampler.sample_calls] == [
        "negative 1",
        "negative 2",
    ]


def test_scale_factor_controls_target_crop_size() -> None:
    """The crop region, not the bbox, controls scaled sampling size."""

    sampler = _FakeSampler()
    segment = _segment(CropRegion(0, 0, 8, 4), BoundingBox(2, 1, 4, 3))
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(segment),
        object(),
        object(),
        [],
        [],
        **(_settings() | {"scale_factor": 2.0, "clamp_size": 0}),
    )

    assert sampler.encoded_shapes == [(1, 8, 16, 3)]


def test_selected_upscale_method_resizes_crop_up() -> None:
    """The user-selected method controls crop upscale only."""

    sampler = _FakeSampler()
    image_resizer = _FakeImageResizer()
    service = _service(sampler, image_resizer)

    service.detail(
        _image(),
        _segs(_segment()),
        object(),
        object(),
        [],
        [],
        **(_settings() | {"upscale_method": "bicubic"}),
    )

    assert image_resizer.upscale_calls == [((1, 4, 4, 3), 8, 8, "bicubic")]


def test_decoded_detail_downscales_with_lanczos() -> None:
    """Decoded detail always uses fixed Lanczos downscaling."""

    sampler = _FakeSampler()
    image_resizer = _FakeImageResizer()
    service = _service(sampler, image_resizer)

    service.detail(
        _image(), _segs(_segment()), object(), object(), [], [], **_settings()
    )

    assert image_resizer.downscale_calls == [((1, 8, 8, 3), 4, 4)]


def test_positive_clamp_limits_target_crop_size() -> None:
    """A positive clamp limits the scaled crop long side."""

    sampler = _FakeSampler()
    segment = _segment(CropRegion(0, 0, 8, 4), BoundingBox(2, 1, 4, 3))
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(segment),
        object(),
        object(),
        [],
        [],
        **(_settings() | {"scale_factor": 4.0, "clamp_size": 12}),
    )

    assert sampler.encoded_shapes == [(1, 6, 12, 3)]


def test_unscaled_crops_are_always_sampled() -> None:
    """Every SEGS region is detailed even when scaling is unnecessary."""

    sampler = _FakeSampler()
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(_segment()),
        object(),
        object(),
        [],
        [],
        **(_settings() | {"scale_factor": 1.0}),
    )

    assert len(sampler.sample_calls) == 1
    assert sampler.encoded_shapes == [(1, 4, 4, 3)]


def test_downscale_factors_sample_at_original_crop_size() -> None:
    """Scale factors below one still inpaint at the crop's original size."""

    sampler = _FakeSampler()
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(_segment()),
        object(),
        object(),
        [],
        [],
        **(_settings() | {"scale_factor": 0.5}),
    )

    assert len(sampler.sample_calls) == 1
    assert sampler.encoded_shapes == [(1, 4, 4, 3)]


def test_noise_mask_true_attaches_latent_mask() -> None:
    """Noise-mask mode passes the crop-local mask to sampling."""

    sampler = _FakeSampler()
    service = _service(sampler)

    service.detail(
        _image(), _segs(_segment()), object(), object(), [], [], **_settings()
    )

    assert sampler.sample_calls[0].noise_mask_shape == (1, 4, 4)


def test_noise_mask_false_samples_without_latent_mask() -> None:
    """Disabling noise_mask omits latent denoise masks."""

    sampler = _FakeSampler()
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(_segment()),
        object(),
        object(),
        [],
        [],
        **(_settings() | {"noise_mask": False}),
    )

    assert sampler.sample_calls[0].noise_mask_shape is None


def test_non_tensor_cropped_mask_is_used_as_crop() -> None:
    """Impact-compatible non-tensor cropped masks are already crop-local."""

    sampler = _FakeSampler()
    segment = _segment(
        CropRegion(4, 4, 8, 8),
        BoundingBox(5, 5, 7, 7),
        cropped_mask=[
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
    )
    service = _service(sampler)

    service.detail(_image(), _segs(segment), object(), object(), [], [], **_settings())

    assert sampler.sample_calls[0].noise_mask_shape == (1, 4, 4)


def test_paste_mask_uses_gaussian_feathered_crop_mask() -> None:
    """Final compositing uses a softened crop-local detailer mask."""

    sampler = _FakeSampler()
    service = _service(sampler)
    mask = torch.zeros((4, 4), dtype=torch.float32)
    mask[1:3, 1:3] = 1.0

    result = service.detail(
        _image(),
        _segs(_segment(cropped_mask=mask)),
        object(),
        object(),
        [],
        [],
        **(_settings() | {"scale_factor": 1.0, "feather": 1}),
    )

    crop = result.image[:, 2:6, 2:6, 0]
    assert 0.0 < float(crop[0, 0, 1]) < 1.0
    assert float(crop[0, 1, 1]) > float(crop[0, 0, 1])
    assert float(crop[0, 3, 3]) < float(crop[0, 1, 1])


def test_noise_mask_feather_is_applied_before_latent_resize() -> None:
    """Noise-mask feathering keeps original crop geometry for ComfyUI resizing."""

    sampler = _FakeSampler()
    service = _service(sampler)
    mask = torch.zeros((4, 4), dtype=torch.float32)
    mask[1:3, 1:3] = 1.0

    service.detail(
        _image(),
        _segs(_segment(cropped_mask=mask)),
        object(),
        object(),
        [],
        [],
        **(_settings() | {"noise_mask_feather": 1}),
    )

    noise_mask = sampler.sample_calls[0].noise_mask
    assert noise_mask is not None
    assert noise_mask.shape == (1, 4, 4)
    assert 0.0 < float(noise_mask[0, 0, 1]) < 1.0
    assert float(noise_mask[0, 1, 1]) > float(noise_mask[0, 0, 1])


def test_noise_mask_feather_applies_model_patch() -> None:
    """Feathered noise masks request differential diffusion patching."""

    sampler = _FakeSampler()
    service = _service(sampler)

    service.detail(
        _image(),
        _segs(_segment()),
        object(),
        object(),
        [],
        [],
        **(_settings() | {"noise_mask_feather": 2}),
    )

    assert sampler.patch_count == 1


def test_batch_images_fail_clearly() -> None:
    """The first detailer version rejects image batches."""

    service = _service(_FakeSampler())

    with pytest.raises(ValueError, match="supports one image at a time"):
        service.detail(
            torch.zeros((2, 8, 8, 3)),
            _segs(_segment()),
            object(),
            object(),
            [],
            [],
            **_settings(),
        )


class _SampleCall:
    """Record one fake sampling call."""

    def __init__(
        self,
        seed: int,
        noise_mask_shape: tuple[int, ...] | None,
        noise_mask: torch.Tensor | None,
        positive: Any,
        negative: Any,
        preview_context: DetailPreviewContext | None,
    ) -> None:
        """Store call fields needed by assertions."""

        self.seed = seed
        self.noise_mask_shape = noise_mask_shape
        self.noise_mask = noise_mask
        self.positive = positive
        self.negative = negative
        self.preview_context = preview_context


class _FakeSampler:
    """Fake VAE/sampling adapter for detailer service tests."""

    def __init__(self) -> None:
        """Create empty call records."""

        self.encoded_shapes: list[tuple[int, ...]] = []
        self.sample_calls: list[_SampleCall] = []
        self.patch_count = 0

    def encode(self, vae: Any, pixels: torch.Tensor, tiled: bool) -> Latent:
        """Record encoded pixel shape and return a small latent."""

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

    def sample(
        self,
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
        preview_context: DetailPreviewContext | None = None,
    ) -> Latent:
        """Record sample options and return the latent unchanged."""

        del model, steps, cfg, sampler_name, scheduler, denoise
        noise_mask = latent_image.get("noise_mask")
        noise_mask_shape = (
            tuple(int(dim) for dim in noise_mask.shape)
            if isinstance(noise_mask, torch.Tensor)
            else None
        )
        self.sample_calls.append(
            _SampleCall(
                seed,
                noise_mask_shape,
                noise_mask.detach().clone()
                if isinstance(noise_mask, torch.Tensor)
                else None,
                positive,
                negative,
                preview_context,
            )
        )
        return latent_image

    def apply_differential_diffusion(self, model: Any) -> Any:
        """Record patching and return the model unchanged."""

        self.patch_count += 1
        return model


class _FakeImageResizer:
    """Fake detail image resizer for service tests."""

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
    sampler: _FakeSampler,
    image_resizer: _FakeImageResizer | None = None,
) -> DetailSEGSByScaleFactorService:
    """Create a service with fake collaborators."""

    return DetailSEGSByScaleFactorService(
        sampler=cast(DetailSamplingBoundary, sampler),
        image_resizer=cast(
            DetailResizeBoundary,
            image_resizer or _FakeImageResizer(),
        ),
    )


def _segs(*segments: Segment) -> NativeSegs:
    """Return native SEGS for an 8x8 image."""

    return (8, 8), tuple(segments)


def _segment(
    crop_region: CropRegion | None = None,
    bbox: BoundingBox | None = None,
    cropped_mask: object | None = None,
) -> Segment:
    """Return one test segment with a cropped mask."""

    resolved_crop_region = crop_region or CropRegion(2, 2, 6, 6)
    resolved_bbox = bbox or BoundingBox(3, 3, 5, 5)
    return Segment(
        cropped_image=None,
        cropped_mask=(
            cropped_mask
            if cropped_mask is not None
            else torch.ones((resolved_crop_region.height, resolved_crop_region.width))
        ),
        confidence=1.0,
        crop_region=resolved_crop_region,
        bbox=resolved_bbox,
        label="face",
    )


def _image() -> torch.Tensor:
    """Return a dark 8x8 image tensor."""

    return torch.zeros((1, 8, 8, 3), dtype=torch.float32)


def _settings() -> dict[str, Any]:
    """Return valid detailer settings for service tests."""

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
    }
