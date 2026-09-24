# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for regional MultiDiffusion sampling runtime."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from simple_syrup.domain.regional_detailing import LatentBox, LatentRegion
from simple_syrup.domain.segs import CropRegion
from simple_syrup.runtime import (
    regional_multidiffusion_sampling,
)
from simple_syrup.runtime.detail_previews import DetailPreviewContext
from simple_syrup.runtime.regional_multidiffusion_prediction import (
    RegionalMultiDiffusionCalcCondBatch,
)

comfy_sample = regional_multidiffusion_sampling._comfy_sample()
comfy_utils = regional_multidiffusion_sampling._comfy_utils()
latent_preview = regional_multidiffusion_sampling._latent_preview()


def _preview_context() -> DetailPreviewContext:
    """Return a minimal regional detail preview context."""

    return DetailPreviewContext(
        image=torch.ones((1, 8, 8, 3), dtype=torch.float32),
        work_region=CropRegion(2, 2, 6, 6),
        work_mask=torch.ones((8, 8), dtype=torch.float32),
        sampled_region=CropRegion(0, 0, 8, 8),
    )


def test_sampling_callback_uses_generic_preview_without_detail_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone regional runtime calls keep generic latent previews."""

    monkeypatch.setattr(
        latent_preview,
        "prepare_callback",
        lambda _model, _steps: "generic callback",
    )

    assert (
        regional_multidiffusion_sampling._sampling_callback(FakeModel(), 4, None)
        == "generic callback"
    )


def test_sampling_callback_uses_detail_preview_with_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detailer regional sampling uses the shared detail preview callback."""

    context = _preview_context()
    calls: dict[str, object] = {}

    def fake_prepare_detail_preview_callback(
        model: FakeModel,
        steps: int,
        preview_context: DetailPreviewContext,
    ) -> str:
        """Record detail preview callback preparation."""

        calls["model"] = model
        calls["steps"] = steps
        calls["preview_context"] = preview_context
        return "detail callback"

    monkeypatch.setattr(
        regional_multidiffusion_sampling,
        "prepare_detail_preview_callback",
        fake_prepare_detail_preview_callback,
    )

    model = FakeModel()
    assert (
        regional_multidiffusion_sampling._sampling_callback(model, 4, context)
        == "detail callback"
    )
    assert calls == {"model": model, "steps": 4, "preview_context": context}


class FakeModel:
    """Provide the ModelPatcher methods used by the runtime."""

    def __init__(
        self,
        model_options: dict[str, Any] | None = None,
        parent: FakeModel | None = None,
    ) -> None:
        """Create a fake model patcher."""

        self.load_device = torch.device("cpu")
        self.model_options = {} if model_options is None else model_options
        self.calc_wrapper: Any = None
        self.model_sampling = object()
        self.parent = parent
        self.clone_count = 0

    def clone(self) -> FakeModel:
        """Return a cloned model with copied options."""

        self.clone_count += 1
        return FakeModel(self.model_options.copy(), parent=self)

    def set_model_sampler_calc_cond_batch_function(self, wrapper: object) -> None:
        """Capture the installed calc-cond-batch wrapper."""

        self.calc_wrapper = wrapper
        self.model_options["sampler_calc_cond_batch_function"] = wrapper

    def set_model_denoise_mask_function(self, denoise_mask_function: object) -> None:
        """Capture the installed denoise-mask function."""

        self.model_options["denoise_mask_function"] = denoise_mask_function

    def get_model_object(self, name: str) -> object:
        """Return the requested fake model object."""

        assert name == "model_sampling"
        return self.model_sampling


class FakeSampler:
    """Represent a resolved sampler in tests."""

    def sample(self, *args: object, **kwargs: object) -> object:
        """Provide ComfyUI's sampler protocol."""

        del args, kwargs
        return None


def test_clone_model_installs_regional_calc_cond_batch_wrapper() -> None:
    """The runtime clones the model and installs a regional wrapper."""

    model = FakeModel()

    wrapped_model, summary = (
        regional_multidiffusion_sampling.clone_model_with_regional_multidiffusion(
            model,
            latent_width=8,
            latent_height=4,
            latent_ndim=4,
            regions=(_region(0, 0, 4, 4, "positive", latent_width=8),),
        )
    )

    assert wrapped_model is not model
    assert isinstance(
        wrapped_model.calc_wrapper,
        RegionalMultiDiffusionCalcCondBatch,
    )
    assert summary.region_count == 1


def test_clone_model_composes_differential_on_same_clone() -> None:
    """Differential diffusion is installed without cloning a temporary parent."""

    model = FakeModel()

    wrapped_model, _summary = (
        regional_multidiffusion_sampling.clone_model_with_regional_multidiffusion(
            model,
            latent_width=8,
            latent_height=4,
            latent_ndim=4,
            regions=(_region(0, 0, 4, 4, "positive", latent_width=8),),
            differential_diffusion=True,
        )
    )

    assert model.clone_count == 1
    assert wrapped_model.parent is model
    assert callable(wrapped_model.model_options["denoise_mask_function"])
    assert isinstance(
        wrapped_model.calc_wrapper,
        RegionalMultiDiffusionCalcCondBatch,
    )


def test_clone_model_rejects_non_callable_existing_calc_wrapper() -> None:
    """Existing calc-cond-batch metadata must be callable."""

    model = FakeModel({"sampler_calc_cond_batch_function": object()})

    with pytest.raises(ValueError, match="Existing sampler_calc_cond_batch_function"):
        regional_multidiffusion_sampling.clone_model_with_regional_multidiffusion(
            model,
            latent_width=8,
            latent_height=4,
            latent_ndim=4,
            regions=(_region(0, 0, 4, 4, "positive", latent_width=8),),
        )


def test_existing_calc_wrapper_is_composed_without_recursion() -> None:
    """Fallback and regional calls delegate to the previous calc wrapper."""

    calls: list[dict[str, Any]] = []

    def existing(args: dict[str, Any]) -> list[torch.Tensor]:
        """Return condition-specific constants and record options."""

        calls.append(args)
        assert args["model_options"].get("sampler_calc_cond_batch_function") is existing
        x = cast(torch.Tensor, args["input"])
        conds = cast(list[object], args["conds"])
        value = 10.0 if _condition_name(conds[0]) == "global" else 20.0
        return [torch.ones_like(x) * value, torch.ones_like(x) * 2.0]

    model = FakeModel(
        {
            "sampler_calc_cond_batch_function": existing,
            "model_function_wrapper": object(),
        }
    )
    wrapped_model, _summary = (
        regional_multidiffusion_sampling.clone_model_with_regional_multidiffusion(
            model,
            latent_width=4,
            latent_height=4,
            latent_ndim=4,
            regions=(_region(0, 0, 4, 4, "regional"),),
        )
    )

    output = wrapped_model.calc_wrapper(
        {
            "conds": ["global", "negative"],
            "input": torch.zeros((1, 1, 4, 4)),
            "sigma": torch.tensor([1.0]),
            "model": wrapped_model,
            "model_options": wrapped_model.model_options,
        }
    )

    assert len(calls) == 2
    assert wrapped_model.model_options["model_function_wrapper"] is not None
    assert torch.allclose(output[0], torch.ones((1, 1, 4, 4)) * 20.0)
    assert torch.allclose(output[1], torch.ones((1, 1, 4, 4)) * 2.0)


def test_shape_mismatch_delegates_to_original_calc_path() -> None:
    """Unexpected model input spatial shapes are delegated unchanged."""

    calls = 0

    def existing(args: dict[str, Any]) -> list[torch.Tensor]:
        """Record fallback calls."""

        nonlocal calls
        calls += 1
        x = cast(torch.Tensor, args["input"])
        return [x + 5.0, x + 1.0]

    model = FakeModel({"sampler_calc_cond_batch_function": existing})
    wrapped_model, _summary = (
        regional_multidiffusion_sampling.clone_model_with_regional_multidiffusion(
            model,
            latent_width=8,
            latent_height=8,
            latent_ndim=4,
            regions=(_region(0, 0, 4, 4, "regional", latent_width=8, latent_height=8),),
        )
    )
    x = torch.zeros((1, 1, 4, 4))

    output = wrapped_model.calc_wrapper(
        {
            "conds": ["global", "negative"],
            "input": x,
            "sigma": torch.tensor([1.0]),
            "model": wrapped_model,
            "model_options": wrapped_model.model_options,
        }
    )

    assert calls == 1
    assert torch.allclose(output[0], x + 5.0)


def test_region_crops_bchw_final_spatial_axes() -> None:
    """Regional calls crop only final height and width axes."""

    calls: list[torch.Tensor] = []

    def existing(args: dict[str, Any]) -> list[torch.Tensor]:
        """Record regional input crops."""

        x = cast(torch.Tensor, args["input"])
        calls.append(x)
        conds = cast(list[object], args["conds"])
        value = 1.0 if _condition_name(conds[0]) == "global" else 3.0
        return [torch.ones_like(x) * value, torch.zeros_like(x)]

    wrapped_model, _summary = (
        regional_multidiffusion_sampling.clone_model_with_regional_multidiffusion(
            FakeModel({"sampler_calc_cond_batch_function": existing}),
            latent_width=8,
            latent_height=4,
            latent_ndim=4,
            regions=(_region(0, 0, 8, 4, "regional", latent_width=8),),
        )
    )
    x = torch.arange(32, dtype=torch.float32).reshape((1, 1, 4, 8))

    output = wrapped_model.calc_wrapper(
        {
            "conds": ["global", "negative"],
            "input": x,
            "sigma": torch.tensor([1.0]),
            "model": wrapped_model,
            "model_options": wrapped_model.model_options,
        }
    )

    assert calls[1].shape == (1, 1, 4, 8)
    assert torch.equal(calls[1], x)
    assert torch.allclose(output[0], torch.ones((1, 1, 4, 8)) * 3.0)


def test_region_crops_singleton_depth_5d_final_spatial_axes() -> None:
    """Anima-style regions crop only final height and width axes."""

    calls: list[torch.Tensor] = []

    def existing(args: dict[str, Any]) -> list[torch.Tensor]:
        """Record regional input crops."""

        x = cast(torch.Tensor, args["input"])
        calls.append(x)
        conds = cast(list[object], args["conds"])
        value = 1.0 if _condition_name(conds[0]) == "global" else 4.0
        return [torch.ones_like(x) * value, torch.zeros_like(x)]

    wrapped_model, _summary = (
        regional_multidiffusion_sampling.clone_model_with_regional_multidiffusion(
            FakeModel({"sampler_calc_cond_batch_function": existing}),
            latent_width=8,
            latent_height=4,
            latent_ndim=5,
            regions=(_region(0, 0, 8, 4, "regional", latent_width=8),),
        )
    )
    x = (
        torch.arange(128, dtype=torch.float32)
        .reshape((1, 16, 1, 4, 2))
        .repeat(1, 1, 1, 1, 4)
    )

    output = wrapped_model.calc_wrapper(
        {
            "conds": ["global", "negative"],
            "input": x,
            "sigma": torch.tensor([1.0]),
            "model": wrapped_model,
            "model_options": wrapped_model.model_options,
        }
    )

    assert calls[1].shape == (1, 16, 1, 4, 8)
    assert torch.equal(calls[1], x)
    assert output[0].shape == x.shape


def test_overlapping_regions_normalize_by_accumulated_weight() -> None:
    """Overlapping regions are averaged before blending over fallback."""

    def existing(args: dict[str, Any]) -> list[torch.Tensor]:
        """Return different constants for each region positive."""

        x = cast(torch.Tensor, args["input"])
        cond = _condition_name(cast(list[object], args["conds"])[0])
        values = {"global": 0.0, "first": 2.0, "second": 6.0}
        return [torch.ones_like(x) * values[cond], torch.zeros_like(x)]

    wrapped_model, _summary = (
        regional_multidiffusion_sampling.clone_model_with_regional_multidiffusion(
            FakeModel({"sampler_calc_cond_batch_function": existing}),
            latent_width=6,
            latent_height=4,
            latent_ndim=4,
            regions=(
                _region(0, 0, 4, 4, "first", latent_width=6),
                _region(2, 0, 4, 4, "second", latent_width=6),
            ),
        )
    )

    output = wrapped_model.calc_wrapper(
        {
            "conds": ["global", "negative"],
            "input": torch.zeros((1, 1, 4, 6)),
            "sigma": torch.tensor([1.0]),
            "model": wrapped_model,
            "model_options": wrapped_model.model_options,
        }
    )

    assert torch.allclose(output[0][:, :, :, :2], torch.ones((1, 1, 4, 2)) * 2.0)
    assert torch.allclose(output[0][:, :, :, 2:4], torch.ones((1, 1, 4, 2)) * 4.0)
    assert torch.allclose(output[0][:, :, :, 4:], torch.ones((1, 1, 4, 2)) * 6.0)


def test_partial_mask_blends_region_over_fallback() -> None:
    """Feathered masks blend region predictions with fallback predictions."""

    def existing(args: dict[str, Any]) -> list[torch.Tensor]:
        """Return fallback or regional constants."""

        x = cast(torch.Tensor, args["input"])
        cond = _condition_name(cast(list[object], args["conds"])[0])
        value = 10.0 if cond == "global" else 20.0
        return [torch.ones_like(x) * value, torch.zeros_like(x)]

    mask = torch.ones((4, 4)) * 0.25
    region = LatentRegion(
        index=0,
        label="soft",
        latent_box=LatentBox(0, 0, 4, 4),
        latent_mask=mask,
        positive=_raw_conditioning("regional"),
    )
    wrapped_model, _summary = (
        regional_multidiffusion_sampling.clone_model_with_regional_multidiffusion(
            FakeModel({"sampler_calc_cond_batch_function": existing}),
            latent_width=4,
            latent_height=4,
            latent_ndim=4,
            regions=(region,),
        )
    )

    output = wrapped_model.calc_wrapper(
        {
            "conds": ["global", "negative"],
            "input": torch.zeros((1, 1, 4, 4)),
            "sigma": torch.tensor([1.0]),
            "model": wrapped_model,
            "model_options": wrapped_model.model_options,
        }
    )

    assert torch.allclose(output[0], torch.ones((1, 1, 4, 4)) * 12.5)


def _region(
    x: int,
    y: int,
    width: int,
    height: int,
    positive: object,
    *,
    latent_width: int = 4,
    latent_height: int = 4,
) -> LatentRegion:
    """Return one full-weight latent region."""

    if isinstance(positive, str):
        positive = _raw_conditioning(positive)
    mask = torch.zeros((latent_height, latent_width), dtype=torch.float32)
    mask[y : y + height, x : x + width] = 1.0
    return LatentRegion(
        index=0,
        label="region",
        latent_box=LatentBox(x, y, width, height),
        latent_mask=mask,
        positive=positive,
    )


def _raw_conditioning(name: str) -> list[list[object]]:
    """Return a raw Comfy CONDITIONING-like value with a visible test name."""

    return [[torch.zeros((1, 1, 1), dtype=torch.float32), {"name": name}]]


def _condition_name(conditioning: object) -> str:
    """Return the test-visible name from raw, processed, or sentinel conditioning."""

    if isinstance(conditioning, str):
        return conditioning
    if isinstance(conditioning, list) and conditioning:
        first = conditioning[0]
        if isinstance(first, dict):
            return str(first.get("name", ""))
        if (
            isinstance(first, list | tuple)
            and len(first) > 1
            and isinstance(
                first[1],
                dict,
            )
        ):
            return str(first[1].get("name", ""))
    return ""
