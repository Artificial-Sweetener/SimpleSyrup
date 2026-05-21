# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Detail SEGS as Regions node contract."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.segs import BoundingBox, CropRegion, NativeSegs, Segment
from simple_syrup.nodes.detail_segs_as_regions import DetailSEGSAsRegions
from simple_syrup.runtime import sampling_samplers, sampling_schedulers
from simple_syrup.services.detail_segs_as_regions_service import (
    DetailSEGSAsRegionsResult,
)


def test_detail_segs_as_regions_node_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regional detailer node exposes its planned ComfyUI contract."""

    monkeypatch.setattr(sampling_samplers, "available_samplers", lambda: ("euler",))
    monkeypatch.setattr(
        sampling_schedulers,
        "available_schedulers",
        lambda: ("normal",),
    )
    inputs = DetailSEGSAsRegions.INPUT_TYPES()

    assert DetailSEGSAsRegions.RETURN_TYPES == ("IMAGE",)
    assert DetailSEGSAsRegions.RETURN_NAMES == ("image",)
    assert DetailSEGSAsRegions.FUNCTION == "detail"
    assert DetailSEGSAsRegions.CATEGORY == "SimpleSyrup/Detailing"
    assert DetailSEGSAsRegions.INPUT_IS_LIST is True
    assert list(inputs["required"]) == [
        "image",
        "model",
        "vae",
        "negative",
        "positive",
        "segs",
        "region_positive",
        "global_prompt_weight",
        "scale_factor",
        "upscale_method",
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
    assert inputs["required"]["region_positive"][0] == "CONDITIONING_BATCH"
    assert inputs["required"]["global_prompt_weight"][1]["default"] == 0.25
    assert inputs["required"]["global_prompt_weight"][1]["min"] == 0.0
    assert inputs["required"]["global_prompt_weight"][1]["max"] == 1.0
    scale_factor_options = inputs["required"]["scale_factor"][1]
    assert scale_factor_options["default"] == 1.0
    assert scale_factor_options["min"] == 1.0
    assert scale_factor_options["max"] == 5.0
    assert scale_factor_options["step"] == 0.1
    assert inputs["required"]["upscale_method"][1]["default"] == "lanczos"
    assert inputs["required"]["noise_mask"][1]["default"] is True


def test_detail_segs_as_regions_node_delegates_all_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regional detailer forwards normalized values to the service."""

    fake_service = _FakeRegionalDetailerService()
    monkeypatch.setattr(
        DetailSEGSAsRegions,
        "service_class",
        staticmethod(lambda: fake_service),
    )
    image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    region_positive = ConditioningBatch((["region"],))

    (output_image,) = DetailSEGSAsRegions().detail(
        image,
        model="model",
        vae="vae",
        negative=["negative"],
        positive=["positive"],
        segs=_segs(),
        region_positive=region_positive,
        global_prompt_weight=0.25,
        scale_factor=2.0,
        upscale_method="bicubic",
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
        tiled_decode=True,
    )

    assert torch.equal(cast(torch.Tensor, output_image), image + 1.0)
    assert len(fake_service.calls) == 1
    call = fake_service.calls[0]
    assert torch.equal(cast(torch.Tensor, call["image"]), image)
    call_segs = cast(NativeSegs, call["segs"])
    assert call_segs[0] == (8, 8)
    assert call_segs[1][0].label == "face"
    assert torch.equal(
        cast(torch.Tensor, call_segs[1][0].cropped_mask), torch.ones((2, 2))
    )
    assert call["model"] == "model"
    assert call["vae"] == "vae"
    assert call["positive"] == ["positive"]
    assert call["negative"] == ["negative"]
    assert call["region_positive"] == region_positive
    assert call["global_prompt_weight"] == 0.25
    assert call["scale_factor"] == 2.0
    assert call["upscale_method"] == "bicubic"
    assert call["seed"] == 1
    assert call["steps"] == 2
    assert call["cfg"] == 7.0
    assert call["sampler_name"] == "euler"
    assert call["scheduler"] == "normal"
    assert call["denoise"] == 0.5
    assert call["feather"] == 5
    assert call["noise_mask"] is True
    assert call["noise_mask_feather"] == 20
    assert call["tiled_encode"] is False
    assert call["tiled_decode"] is True


def test_detail_segs_as_regions_node_pairs_image_batch_segs_and_region_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """List-mode region_positive values are paired one-to-one with SEGS."""

    fake_service = _FakeRegionalDetailerService()
    monkeypatch.setattr(
        DetailSEGSAsRegions,
        "service_class",
        staticmethod(lambda: fake_service),
    )
    image = torch.zeros((2, 8, 8, 3), dtype=torch.float32)
    first_region_positive = ConditioningBatch((["first"],))
    second_region_positive = ConditioningBatch((["second"],))

    (output_image,) = DetailSEGSAsRegions().detail(
        image=[image],
        model=["model"],
        vae=["vae"],
        negative=[["negative"]],
        positive=[["positive"]],
        segs=[_segs("first"), _segs("second")],
        region_positive=[first_region_positive, second_region_positive],
        global_prompt_weight=[0.4],
        scale_factor=[2.0],
        upscale_method=["area"],
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
        tiled_decode=[True],
    )

    assert torch.equal(cast(torch.Tensor, output_image), image + 1.0)
    assert [call["region_positive"] for call in fake_service.calls] == [
        first_region_positive,
        second_region_positive,
    ]
    labels = [cast(NativeSegs, call["segs"])[1][0].label for call in fake_service.calls]
    assert labels == ["first", "second"]


def test_detail_segs_as_regions_node_rejects_broadcast_region_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-image list mode requires one regional batch per SEGS payload."""

    monkeypatch.setattr(
        DetailSEGSAsRegions,
        "service_class",
        _FakeRegionalDetailerService,
    )

    with pytest.raises(ValueError, match="received 1 conditioning batches for 2"):
        DetailSEGSAsRegions().detail(
            image=[torch.zeros((2, 8, 8, 3), dtype=torch.float32)],
            model=["model"],
            vae=["vae"],
            negative=[["negative"]],
            positive=[["positive"]],
            segs=[_segs("first"), _segs("second")],
            region_positive=[ConditioningBatch((["only"],))],
            global_prompt_weight=[0.25],
            scale_factor=[1.0],
            upscale_method=["lanczos"],
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
            tiled_decode=[True],
        )


def test_detail_segs_as_regions_node_rejects_image_segs_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regional detailer mirrors existing image and SEGS pairing validation."""

    monkeypatch.setattr(
        DetailSEGSAsRegions,
        "service_class",
        _FakeRegionalDetailerService,
    )

    with pytest.raises(ValueError, match="received 2 images and 1 SEGS payload"):
        DetailSEGSAsRegions().detail(
            image=[torch.zeros((2, 8, 8, 3), dtype=torch.float32)],
            model=["model"],
            vae=["vae"],
            negative=[["negative"]],
            positive=[["positive"]],
            segs=[_segs()],
            region_positive=[ConditioningBatch((["region"],))],
            global_prompt_weight=[0.25],
            scale_factor=[1.0],
            upscale_method=["lanczos"],
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
            tiled_decode=[True],
        )


class _FakeRegionalDetailerService:
    """Fake regional detailer service for node tests."""

    def __init__(self) -> None:
        """Create a call-recording fake service."""

        self.calls: list[dict[str, object]] = []

    def detail(
        self,
        image: object,
        segs: object,
        model: Any,
        vae: Any,
        positive: Any,
        negative: Any,
        region_positive: object,
        global_prompt_weight: float,
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
    ) -> DetailSEGSAsRegionsResult:
        """Return deterministic detailer output and record the call."""

        self.calls.append(
            {
                "image": image,
                "segs": segs,
                "model": model,
                "vae": vae,
                "positive": positive,
                "negative": negative,
                "region_positive": region_positive,
                "global_prompt_weight": global_prompt_weight,
                "scale_factor": scale_factor,
                "upscale_method": upscale_method,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "feather": feather,
                "noise_mask": noise_mask,
                "noise_mask_feather": noise_mask_feather,
                "tiled_encode": tiled_encode,
                "tiled_decode": tiled_decode,
            }
        )
        return DetailSEGSAsRegionsResult(image=cast(torch.Tensor, image) + 1.0)


def _segs(label: str = "face") -> NativeSegs:
    """Return native SEGS for node tests."""

    segment = Segment(
        cropped_image=None,
        cropped_mask=torch.ones((2, 2)),
        confidence=1.0,
        crop_region=CropRegion(0, 0, 2, 2),
        bbox=BoundingBox(0, 0, 2, 2),
        label=label,
    )
    return (8, 8), (segment,)
