# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for regional MultiDiffusion sampling runtime."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.regional_detailing import LatentBox, LatentRegion
from simple_syrup.runtime import (
    regional_multidiffusion_sampling,
    sampling_samplers,
    sampling_schedulers,
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
