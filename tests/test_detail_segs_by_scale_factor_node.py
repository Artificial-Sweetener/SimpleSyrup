# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Detail SEGS by Scale Factor node contract."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from simple_syrup.domain.segs import BoundingBox, CropRegion, NativeSegs, Segment
from simple_syrup.nodes.detail_segs_by_scale_factor import DetailSEGSByScaleFactor
from simple_syrup.runtime import sampling_samplers, sampling_schedulers
from simple_syrup.services.detail_segs_by_scale_factor_service import DetailerResult


def test_detail_segs_by_scale_factor_node_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detailer node exposes planned inputs, outputs, and category."""

    monkeypatch.setattr(sampling_samplers, "available_samplers", lambda: ("euler",))
    monkeypatch.setattr(
        sampling_schedulers, "available_schedulers", lambda: ("normal",)
    )
    inputs = DetailSEGSByScaleFactor.INPUT_TYPES()

    assert DetailSEGSByScaleFactor.RETURN_TYPES == ("IMAGE",)
    assert DetailSEGSByScaleFactor.RETURN_NAMES == ("image",)
    assert DetailSEGSByScaleFactor.INPUT_IS_LIST is True
    assert DetailSEGSByScaleFactor.CATEGORY == "SimpleSyrup/Detailing"
    assert list(inputs["required"]) == [
        "image",
        "segs",
        "model",
        "vae",
        "positive",
        "negative",
        "scale_factor",
        "upscale_method",
        "clamp_size",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "denoise",
        "feather",
        "noise_mask",
        "noise_mask_feather",
        "tiled_encode",
        "tiled_decode",
    ]
    assert inputs["required"]["positive"][0] == "CONDITIONING,CONDITIONING_BATCH"
    assert inputs["required"]["negative"][0] == "CONDITIONING,CONDITIONING_BATCH"
    assert inputs["required"]["upscale_method"][0] == [
        "nearest-exact",
        "bilinear",
        "area",
        "bicubic",
        "lanczos",
    ]
    scale_factor_options = inputs["required"]["scale_factor"][1]
    assert scale_factor_options["default"] == 1.5
    assert scale_factor_options["min"] == 1.0
    assert scale_factor_options["max"] == 5.0
    assert scale_factor_options["step"] == 0.1
    assert inputs["required"]["upscale_method"][1]["default"] == "lanczos"
    assert inputs["required"]["noise_mask_feather"][1]["default"] == 20
    assert "preview_mode" not in inputs["required"]


def test_detail_segs_by_scale_factor_node_returns_only_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detailer node emits only the detailed image."""

    monkeypatch.setattr(
        DetailSEGSByScaleFactor,
        "service_class",
        _FakeDetailerService,
    )
    image = torch.zeros((1, 8, 8, 3))

    (output_image,) = DetailSEGSByScaleFactor().detail(
        image,
        _segs(),
        object(),
        object(),
        [],
        [],
        scale_factor=1.5,
        upscale_method="bicubic",
        clamp_size=0,
        seed=1,
        steps=2,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        denoise=0.5,
        feather=5,
        noise_mask=True,
        noise_mask_feather=20,
        tiled_encode=False,
        tiled_decode=False,
    )

    assert torch.equal(cast(torch.Tensor, output_image), image + 1.0)


def test_detail_segs_by_scale_factor_node_accepts_image_batch_and_segs_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detailer pairs each image in a batch with its matching SEGS payload."""

    monkeypatch.setattr(
        DetailSEGSByScaleFactor,
        "service_class",
        _FakeDetailerService,
    )
    image = torch.zeros((2, 8, 8, 3), dtype=torch.float32)

    (output_image,) = DetailSEGSByScaleFactor().detail(
        image=[image],
        segs=[_segs(), _segs()],
        model=[object()],
        vae=[object()],
        positive=[[]],
        negative=[[]],
        scale_factor=[1.5],
        upscale_method=["bicubic"],
        clamp_size=[0],
        seed=[1],
        steps=[2],
        cfg=[7.0],
        sampler_name=["euler"],
        scheduler=["normal"],
        denoise=[0.5],
        feather=[5],
        noise_mask=[True],
        noise_mask_feather=[20],
        tiled_encode=[False],
        tiled_decode=[False],
    )

    assert torch.equal(cast(torch.Tensor, output_image), image + 1.0)


def test_detail_segs_by_scale_factor_node_rejects_batch_image_with_single_segs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch image needs one SEGS payload per image."""

    monkeypatch.setattr(
        DetailSEGSByScaleFactor,
        "service_class",
        _FakeDetailerService,
    )

    with pytest.raises(ValueError, match="received 2 images and 1 SEGS payload"):
        DetailSEGSByScaleFactor().detail(
            torch.zeros((2, 8, 8, 3), dtype=torch.float32),
            _segs(),
            object(),
            object(),
            [],
            [],
            scale_factor=1.5,
            upscale_method="lanczos",
            clamp_size=0,
            seed=1,
            steps=2,
            cfg=7.0,
            sampler_name="euler",
            scheduler="normal",
            denoise=0.5,
            feather=5,
            noise_mask=True,
            noise_mask_feather=20,
            tiled_encode=False,
            tiled_decode=False,
        )


def test_detail_segs_by_scale_factor_node_rejects_mismatched_segs_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image and SEGS list lengths must match exactly."""

    monkeypatch.setattr(
        DetailSEGSByScaleFactor,
        "service_class",
        _FakeDetailerService,
    )

    with pytest.raises(ValueError, match="received 2 images and 1 SEGS payload"):
        DetailSEGSByScaleFactor().detail(
            image=[torch.zeros((2, 8, 8, 3), dtype=torch.float32)],
            segs=[_segs()],
            model=[object()],
            vae=[object()],
            positive=[[]],
            negative=[[]],
            scale_factor=[1.5],
            upscale_method=["lanczos"],
            clamp_size=[0],
            seed=[1],
            steps=[2],
            cfg=[7.0],
            sampler_name=["euler"],
            scheduler=["normal"],
            denoise=[0.5],
            feather=[5],
            noise_mask=[True],
            noise_mask_feather=[20],
            tiled_encode=[False],
            tiled_decode=[False],
        )


def test_detail_segs_by_scale_factor_node_rejects_single_image_with_segs_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple SEGS payloads cannot target one image."""

    monkeypatch.setattr(
        DetailSEGSByScaleFactor,
        "service_class",
        _FakeDetailerService,
    )

    with pytest.raises(ValueError, match="received 1 images and 2 SEGS payloads"):
        DetailSEGSByScaleFactor().detail(
            image=[torch.zeros((1, 8, 8, 3), dtype=torch.float32)],
            segs=[_segs(), _segs()],
            model=[object()],
            vae=[object()],
            positive=[[]],
            negative=[[]],
            scale_factor=[1.5],
            upscale_method=["lanczos"],
            clamp_size=[0],
            seed=[1],
            steps=[2],
            cfg=[7.0],
            sampler_name=["euler"],
            scheduler=["normal"],
            denoise=[0.5],
            feather=[5],
            noise_mask=[True],
            noise_mask_feather=[20],
            tiled_encode=[False],
            tiled_decode=[False],
        )


class _FakeDetailerService:
    """Fake detailer service for node tests."""

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
    ) -> DetailerResult:
        """Return deterministic detailer output."""

        del segs, model, vae, positive, negative, scale_factor
        del upscale_method, clamp_size, seed, steps, cfg
        del sampler_name, scheduler, denoise, feather, noise_mask
        del noise_mask_feather, tiled_encode, tiled_decode
        return DetailerResult(image=cast(torch.Tensor, image) + 1.0)


def _segs() -> NativeSegs:
    """Return native SEGS for node tests."""

    segment = Segment(
        cropped_image=None,
        cropped_mask=torch.ones((2, 2)),
        confidence=1.0,
        crop_region=CropRegion(0, 0, 2, 2),
        bbox=BoundingBox(0, 0, 2, 2),
        label="face",
    )
    return (8, 8), (segment,)
