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
from simple_syrup.runtime import (
    regional_multidiffusion_prediction,
    regional_multidiffusion_sampling,
)

comfy_sample = regional_multidiffusion_sampling._comfy_sample()
comfy_utils = regional_multidiffusion_sampling._comfy_utils()
latent_preview = regional_multidiffusion_sampling._latent_preview()


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
        regional_multidiffusion_prediction,
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
