# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for regional MultiDiffusion sampling runtime."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from simple_syrup.domain.regional_detailing import LatentBox, LatentRegion
from simple_syrup.domain.segs import CropRegion
from simple_syrup.runtime import (
    regional_multidiffusion_sampling,
    sampling_samplers,
    sampling_schedulers,
)
from simple_syrup.runtime.detail_previews import DetailPreviewContext

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
    ) -> None:
        """Create a fake model patcher."""

        self.load_device = torch.device("cpu")
        self.model_options = {} if model_options is None else model_options
        self.calc_wrapper: Any = None
        self.model_sampling = object()

    def clone(self) -> FakeModel:
        """Return a cloned model with copied options."""

        return FakeModel(self.model_options.copy())

    def set_model_sampler_calc_cond_batch_function(self, wrapper: object) -> None:
        """Capture the installed calc-cond-batch wrapper."""

        self.calc_wrapper = wrapper
        self.model_options["sampler_calc_cond_batch_function"] = wrapper

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
        regional_multidiffusion_sampling.RegionalMultiDiffusionCalcCondBatch,
    )
    assert summary.region_count == 1


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


def test_raw_region_conditioning_is_converted_before_calc_cond_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw CONDITIONING entries are converted before Comfy calc-cond-batch calls."""

    calls: list[list[list[dict[str, Any]] | None]] = []

    def calc_cond_batch(
        _model: object,
        conds: list[list[dict[str, Any]] | None],
        x_in: torch.Tensor,
        _timestep: torch.Tensor,
        _model_options: dict[str, Any],
    ) -> list[torch.Tensor]:
        """Return constants while asserting sampler-ready conditioning shape."""

        calls.append(conds)
        first_cond = conds[0]
        assert first_cond is not None
        assert isinstance(first_cond[0], dict)
        assert "model_conds" in first_cond[0]
        value = 5.0 if "cross_attn" in first_cond[0] else 1.0
        return [torch.ones_like(x_in) * value, torch.zeros_like(x_in)]

    fake_samplers = SimpleNamespace(
        calc_cond_batch=calc_cond_batch,
        resolve_areas_and_cond_masks_multidim=lambda *_args: None,
        calculate_start_end_timesteps=lambda *_args: None,
    )
    monkeypatch.setattr(
        regional_multidiffusion_sampling,
        "_comfy_samplers",
        lambda: fake_samplers,
    )
    raw_region_positive = [[torch.ones((1, 1, 1)), {}]]
    wrapped_model, _summary = (
        regional_multidiffusion_sampling.clone_model_with_regional_multidiffusion(
            FakeModel(),
            latent_width=4,
            latent_height=4,
            latent_ndim=4,
            regions=(_region(0, 0, 4, 4, raw_region_positive),),
        )
    )

    output = wrapped_model.calc_wrapper(
        {
            "conds": [
                [{"model_conds": {}, "uuid": object()}],
                [{"model_conds": {}, "uuid": object()}],
            ],
            "input": torch.zeros((1, 1, 4, 4)),
            "sigma": torch.tensor([1.0]),
            "model": wrapped_model,
            "model_options": wrapped_model.model_options,
        }
    )

    assert len(calls) == 2
    assert "cross_attn" in cast(list[dict[str, Any]], calls[1][0])[0]
    assert torch.allclose(output[0], torch.ones((1, 1, 4, 4)) * 5.0)


def test_global_prompt_weight_blends_full_region_with_global_prediction() -> None:
    """Covered pixels keep the configured global positive prediction share."""

    def existing(args: dict[str, Any]) -> list[torch.Tensor]:
        """Return fallback or regional constants."""

        x = cast(torch.Tensor, args["input"])
        cond = _condition_name(cast(list[object], args["conds"])[0])
        value = 10.0 if cond == "global" else 20.0
        return [torch.ones_like(x) * value, torch.zeros_like(x)]

    wrapped_model, _summary = (
        regional_multidiffusion_sampling.clone_model_with_regional_multidiffusion(
            FakeModel({"sampler_calc_cond_batch_function": existing}),
            latent_width=4,
            latent_height=4,
            latent_ndim=4,
            regions=(_region(0, 0, 4, 4, "regional"),),
            global_prompt_weight=0.25,
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

    assert torch.allclose(output[0], torch.ones((1, 1, 4, 4)) * 17.5)


def test_global_prompt_weight_keeps_partial_mask_coverage() -> None:
    """Soft masks scale regional influence before global/regional weighting."""

    def existing(args: dict[str, Any]) -> list[torch.Tensor]:
        """Return fallback or regional constants."""

        x = cast(torch.Tensor, args["input"])
        cond = _condition_name(cast(list[object], args["conds"])[0])
        value = 10.0 if cond == "global" else 20.0
        return [torch.ones_like(x) * value, torch.zeros_like(x)]

    region = LatentRegion(
        index=0,
        label="soft",
        latent_box=LatentBox(0, 0, 4, 4),
        latent_mask=torch.ones((4, 4)) * 0.5,
        positive=_raw_conditioning("regional"),
    )
    wrapped_model, _summary = (
        regional_multidiffusion_sampling.clone_model_with_regional_multidiffusion(
            FakeModel({"sampler_calc_cond_batch_function": existing}),
            latent_width=4,
            latent_height=4,
            latent_ndim=4,
            regions=(region,),
            global_prompt_weight=0.25,
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

    assert torch.allclose(output[0], torch.ones((1, 1, 4, 4)) * 13.75)


def test_overlapping_regions_normalize_before_global_prompt_weight_blend() -> None:
    """Overlaps average region predictions before applying global weight."""

    def existing(args: dict[str, Any]) -> list[torch.Tensor]:
        """Return different constants for each condition."""

        x = cast(torch.Tensor, args["input"])
        cond = _condition_name(cast(list[object], args["conds"])[0])
        values = {"global": 10.0, "first": 20.0, "second": 40.0}
        return [torch.ones_like(x) * values[cond], torch.zeros_like(x)]

    wrapped_model, _summary = (
        regional_multidiffusion_sampling.clone_model_with_regional_multidiffusion(
            FakeModel({"sampler_calc_cond_batch_function": existing}),
            latent_width=4,
            latent_height=4,
            latent_ndim=4,
            regions=(
                _region(0, 0, 4, 4, "first"),
                _region(0, 0, 4, 4, "second"),
            ),
            global_prompt_weight=0.25,
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

    assert torch.allclose(output[0], torch.ones((1, 1, 4, 4)) * 25.0)


def test_negative_conditioning_is_reused_for_region_unconditional_path() -> None:
    """Regional calls keep the original negative conditioning."""

    regional_conds: list[list[object]] = []

    def existing(args: dict[str, Any]) -> list[torch.Tensor]:
        """Record all conditioning lists."""

        x = cast(torch.Tensor, args["input"])
        conds = cast(list[object], args["conds"])
        regional_conds.append(conds)
        return [torch.ones_like(x), torch.ones_like(x) * 2.0]

    wrapped_model, _summary = (
        regional_multidiffusion_sampling.clone_model_with_regional_multidiffusion(
            FakeModel({"sampler_calc_cond_batch_function": existing}),
            latent_width=4,
            latent_height=4,
            latent_ndim=4,
            regions=(_region(0, 0, 4, 4, "regional"),),
        )
    )

    wrapped_model.calc_wrapper(
        {
            "conds": ["global", "negative"],
            "input": torch.zeros((1, 1, 4, 4)),
            "sigma": torch.tensor([1.0]),
            "model": wrapped_model,
            "model_options": wrapped_model.model_options,
        }
    )

    assert [_condition_name(item) for item in regional_conds[1]] == [
        "regional",
        "negative",
    ]


def test_cfg_one_none_uncond_still_returns_two_entries() -> None:
    """Comfy's CFG=1 optimization keeps the output list shape stable."""

    def existing(args: dict[str, Any]) -> list[torch.Tensor]:
        """Return one tensor per cond slot, even when uncond is None."""

        x = cast(torch.Tensor, args["input"])
        conds = cast(list[object], args["conds"])
        value = 3.0 if _condition_name(conds[0]) == "regional" else 1.0
        return [torch.ones_like(x) * value, torch.zeros_like(x)]

    wrapped_model, _summary = (
        regional_multidiffusion_sampling.clone_model_with_regional_multidiffusion(
            FakeModel({"sampler_calc_cond_batch_function": existing}),
            latent_width=4,
            latent_height=4,
            latent_ndim=4,
            regions=(_region(0, 0, 4, 4, "regional"),),
        )
    )

    output = wrapped_model.calc_wrapper(
        {
            "conds": ["global", None],
            "input": torch.zeros((1, 1, 4, 4)),
            "sigma": torch.tensor([1.0]),
            "model": wrapped_model,
            "model_options": wrapped_model.model_options,
        }
    )

    assert len(output) == 2
    assert torch.allclose(output[0], torch.ones((1, 1, 4, 4)) * 3.0)
    assert torch.allclose(output[1], torch.zeros((1, 1, 4, 4)))


def test_sample_rejects_unipc_before_sampler_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UniPC sampler names fail before ComfyUI sampler lookup."""

    def fail_resolve_sampler(_sampler_name: str) -> FakeSampler:
        """Fail if sampler resolution is reached."""

        raise AssertionError("UniPC rejection should happen before sampler resolution")

    monkeypatch.setattr(sampling_samplers, "resolve_sampler", fail_resolve_sampler)

    with pytest.raises(ValueError, match="not compatible with UniPC"):
        regional_multidiffusion_sampling.sample_regional_multidiffusion(
            model=FakeModel(),
            seed=1,
            steps=1,
            cfg=1.0,
            sampler_name="uni_pc",
            scheduler="normal",
            positive=[],
            negative=[],
            latent_image={"samples": torch.zeros((1, 4, 4, 4))},
            regions=(_region(0, 0, 4, 4, "regional"),),
            denoise=1.0,
            global_prompt_weight=0.0,
        )


def test_sample_rejects_unsupported_conditioning() -> None:
    """Regional and ControlNet conditioning fail closed."""

    with pytest.raises(ValueError, match="regional conditioning or ControlNet"):
        regional_multidiffusion_sampling.sample_regional_multidiffusion(
            model=FakeModel(),
            seed=1,
            steps=1,
            cfg=1.0,
            sampler_name="euler",
            scheduler="normal",
            positive=[{"area": (4, 4, 0, 0)}],
            negative=[],
            latent_image={"samples": torch.zeros((1, 4, 4, 4))},
            regions=(_region(0, 0, 4, 4, "regional"),),
            denoise=1.0,
            global_prompt_weight=0.0,
        )


def test_sample_delegates_to_comfy_sampling_with_cloned_wrapped_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling mirrors KSampler flow while using a wrapped model clone."""

    calls: dict[str, Any] = {}
    model = FakeModel()
    sampler = FakeSampler()
    latent_samples = torch.zeros((1, 4, 4, 8), dtype=torch.float32)
    fixed_noise = torch.ones_like(latent_samples)
    fixed_sigmas = torch.tensor([1.0, 0.0], dtype=torch.float32)
    sampled = torch.full_like(latent_samples, 0.25)
    latent_image: dict[str, Any] = {
        "samples": latent_samples,
        "downscale_ratio_spacial": 2,
        "kept": "value",
    }

    monkeypatch.setattr(
        sampling_samplers,
        "resolve_sampler",
        lambda _sampler_name: sampler,
    )
    monkeypatch.setattr(
        sampling_schedulers,
        "calculate_sigmas",
        lambda **_kwargs: fixed_sigmas,
    )
    monkeypatch.setattr(
        comfy_sample,
        "fix_empty_latent_channels",
        lambda _model, samples, _downscale_ratio_spacial: samples,
    )
    monkeypatch.setattr(
        comfy_sample,
        "prepare_noise",
        lambda samples, _seed, _batch_inds=None: fixed_noise,
    )
    monkeypatch.setattr(latent_preview, "prepare_callback", lambda _model, _steps: None)
    monkeypatch.setattr(comfy_utils, "PROGRESS_BAR_ENABLED", False)

    def fake_sample_custom(
        received_model: FakeModel,
        noise: torch.Tensor,
        cfg: float,
        received_sampler: FakeSampler,
        sigmas: torch.Tensor,
        positive: object,
        negative: object,
        latent_image: torch.Tensor,
        noise_mask: torch.Tensor | None,
        callback: object,
        disable_pbar: bool,
        seed: int,
    ) -> torch.Tensor:
        """Record sample_custom arguments."""

        calls["sample_custom"] = {
            "model": received_model,
            "noise": noise,
            "cfg": cfg,
            "sampler": received_sampler,
            "sigmas": sigmas,
            "positive": positive,
            "negative": negative,
            "latent_image": latent_image,
            "noise_mask": noise_mask,
            "callback": callback,
            "disable_pbar": disable_pbar,
            "seed": seed,
        }
        return sampled

    monkeypatch.setattr(comfy_sample, "sample_custom", fake_sample_custom)

    output = regional_multidiffusion_sampling.sample_regional_multidiffusion(
        model=model,
        seed=123,
        steps=2,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        positive=[{"model_conds": {}}],
        negative=[{"model_conds": {}}],
        latent_image=latent_image,
        regions=(_region(0, 0, 4, 4, [{"model_conds": {}}], latent_width=8),),
        denoise=1.0,
        global_prompt_weight=0.25,
    )

    assert output is not latent_image
    assert output["samples"] is sampled
    assert output["kept"] == "value"
    assert "downscale_ratio_spacial" not in output
    assert calls["sample_custom"]["model"] is not model
    assert calls["sample_custom"]["model"].calc_wrapper is not None
    assert calls["sample_custom"]["sampler"] is sampler
    assert calls["sample_custom"]["noise"] is fixed_noise
    assert calls["sample_custom"]["disable_pbar"] is True


def test_sample_accepts_singleton_depth_5d_latent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anima-style singleton-depth latents pass runtime validation."""

    def fake_sample_custom(
        _model: object,
        _noise: torch.Tensor,
        _cfg: float,
        _sampler: object,
        _sigmas: torch.Tensor,
        _positive: object,
        _negative: object,
        latent_image: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        """Return a deterministic sampled latent for shape validation."""

        return latent_image + 1.0

    model = FakeModel()
    sampler = FakeSampler()
    latent_samples = torch.zeros((1, 16, 1, 4, 8), dtype=torch.float32)
    fixed_sigmas = torch.tensor([1.0, 0.0], dtype=torch.float32)

    monkeypatch.setattr(
        sampling_samplers,
        "resolve_sampler",
        lambda _sampler_name: sampler,
    )
    monkeypatch.setattr(
        sampling_schedulers,
        "calculate_sigmas",
        lambda **_kwargs: fixed_sigmas,
    )
    monkeypatch.setattr(
        comfy_sample,
        "fix_empty_latent_channels",
        lambda _model, samples, _downscale_ratio_spacial: samples,
    )
    monkeypatch.setattr(
        comfy_sample,
        "prepare_noise",
        lambda samples, _seed, _batch_inds=None: torch.ones_like(samples),
    )
    monkeypatch.setattr(latent_preview, "prepare_callback", lambda _model, _steps: None)
    monkeypatch.setattr(comfy_utils, "PROGRESS_BAR_ENABLED", True)
    monkeypatch.setattr(comfy_sample, "sample_custom", fake_sample_custom)

    output = regional_multidiffusion_sampling.sample_regional_multidiffusion(
        model=model,
        seed=123,
        steps=2,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        positive=[{"model_conds": {}}],
        negative=[{"model_conds": {}}],
        latent_image={"samples": latent_samples},
        regions=(_region(0, 0, 4, 4, [{"model_conds": {}}], latent_width=8),),
        denoise=1.0,
        global_prompt_weight=0.25,
    )

    assert torch.equal(output["samples"], latent_samples + 1.0)


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
