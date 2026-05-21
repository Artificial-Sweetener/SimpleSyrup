# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for MultiDiffusion ComfyUI sampling runtime."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from simple_syrup.domain.segs import CropRegion
from simple_syrup.runtime import (
    multidiffusion_sampling,
    sampling_samplers,
    sampling_schedulers,
)
from simple_syrup.runtime.detail_previews import DetailPreviewContext

comfy_sample = multidiffusion_sampling._comfy_sample()
comfy_utils = multidiffusion_sampling._comfy_utils()
latent_preview = multidiffusion_sampling._latent_preview()


def _preview_context() -> DetailPreviewContext:
    """Return a minimal detail preview context for runtime callback tests."""

    return DetailPreviewContext(
        image=torch.ones((1, 8, 8, 3), dtype=torch.float32),
        work_region=CropRegion(2, 2, 6, 6),
        work_mask=torch.ones((4, 4), dtype=torch.float32),
    )


def test_sampling_callback_uses_generic_preview_without_detail_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KSampler-style MultiDiffusion keeps the generic latent preview path."""

    monkeypatch.setattr(
        latent_preview,
        "prepare_callback",
        lambda _model, _steps: "generic callback",
    )

    assert (
        multidiffusion_sampling._sampling_callback(FakeModel(), 4, None)
        == "generic callback"
    )


def test_sampling_callback_uses_detail_preview_with_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detailer MultiDiffusion uses the shared detail preview callback."""

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
        multidiffusion_sampling,
        "prepare_detail_preview_callback",
        fake_prepare_detail_preview_callback,
    )

    model = FakeModel()
    assert (
        multidiffusion_sampling._sampling_callback(model, 4, context)
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
        self.wrapper: Any = None
        self.model_sampling = object()

    def clone(self) -> FakeModel:
        """Return a cloned model with copied options."""

        return FakeModel(self.model_options.copy())

    def set_model_unet_function_wrapper(self, wrapper: object) -> None:
        """Capture the installed model function wrapper."""

        self.wrapper = wrapper
        self.model_options["model_function_wrapper"] = wrapper

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


def test_clone_model_with_multidiffusion_installs_wrapper() -> None:
    """The runtime clones the model and installs a MultiDiffusion wrapper."""

    model = FakeModel()

    wrapped_model, plan = multidiffusion_sampling.clone_model_with_multidiffusion(
        model,
        latent_width=8,
        latent_height=4,
        tile_width=4,
        tile_height=4,
        overlap=0,
        tile_batch_size=2,
    )

    assert wrapped_model is not model
    assert isinstance(
        wrapped_model.wrapper,
        multidiffusion_sampling.MultiDiffusionModelWrapper,
    )
    assert plan.tile_batch_size == 2


def test_clone_model_rejects_non_callable_existing_wrapper() -> None:
    """Existing wrapper metadata must be callable."""

    model = FakeModel({"model_function_wrapper": object()})

    with pytest.raises(ValueError, match="Existing model_function_wrapper"):
        multidiffusion_sampling.clone_model_with_multidiffusion(
            model,
            latent_width=8,
            latent_height=4,
            tile_width=4,
            tile_height=4,
            overlap=0,
            tile_batch_size=1,
        )


def test_model_wrapper_preserves_existing_wrapper() -> None:
    """Existing model_function_wrapper is composed for each tile call."""

    wrapper_calls = 0

    def old_wrapper(apply_model: object, args: dict[str, object]) -> torch.Tensor:
        """Record old wrapper calls and delegate."""

        del apply_model
        nonlocal wrapper_calls
        wrapper_calls += 1
        return cast(torch.Tensor, args["input"]) + 10.0

    model = FakeModel({"model_function_wrapper": old_wrapper})
    wrapped_model, _plan = multidiffusion_sampling.clone_model_with_multidiffusion(
        model,
        latent_width=8,
        latent_height=4,
        tile_width=4,
        tile_height=4,
        overlap=0,
        tile_batch_size=1,
    )

    output = wrapped_model.wrapper(
        lambda x, timestep, **c: x,
        {
            "input": torch.zeros((1, 1, 4, 8)),
            "timestep": torch.tensor([1.0]),
            "c": {},
            "cond_or_uncond": [0],
        },
    )

    assert wrapper_calls == 2
    assert torch.allclose(output, torch.ones((1, 1, 4, 8)) * 10.0)


def test_model_wrapper_delegates_shape_mismatch_unchanged() -> None:
    """Unexpected model input spatial shapes are delegated without tiling."""

    model = FakeModel()
    wrapped_model, _plan = multidiffusion_sampling.clone_model_with_multidiffusion(
        model,
        latent_width=8,
        latent_height=4,
        tile_width=4,
        tile_height=4,
        overlap=0,
        tile_batch_size=1,
    )
    calls = 0

    def apply_model(
        x: torch.Tensor,
        timestep: torch.Tensor,
        **c: object,
    ) -> torch.Tensor:
        """Record direct fallback calls."""

        del timestep, c
        nonlocal calls
        calls += 1
        return x + 5.0

    x = torch.zeros((1, 1, 2, 2))
    output = wrapped_model.wrapper(
        apply_model,
        {"input": x, "timestep": torch.tensor([1.0]), "c": {}, "cond_or_uncond": [0]},
    )

    assert calls == 1
    assert torch.allclose(output, x + 5.0)


def test_model_wrapper_delegates_single_tile_plan_unchanged() -> None:
    """Plans with one tile preserve the reference extension's no-op behavior."""

    model = FakeModel()
    wrapped_model, _plan = multidiffusion_sampling.clone_model_with_multidiffusion(
        model,
        latent_width=4,
        latent_height=4,
        tile_width=4,
        tile_height=4,
        overlap=0,
        tile_batch_size=1,
    )
    calls = 0

    def apply_model(
        x: torch.Tensor,
        timestep: torch.Tensor,
        **c: object,
    ) -> torch.Tensor:
        """Record direct fallback calls."""

        del timestep, c
        nonlocal calls
        calls += 1
        return x + 2.0

    x = torch.zeros((1, 1, 4, 4))
    output = wrapped_model.wrapper(
        apply_model,
        {"input": x, "timestep": torch.tensor([1.0]), "c": {}, "cond_or_uncond": [0]},
    )

    assert calls == 1
    assert torch.allclose(output, x + 2.0)


def test_model_wrapper_tiles_input_conditioning_and_transformer_options() -> None:
    """The wrapper tiles latents, conditioning tensors, timesteps, and metadata."""

    model = FakeModel()
    wrapped_model, _plan = multidiffusion_sampling.clone_model_with_multidiffusion(
        model,
        latent_width=8,
        latent_height=4,
        tile_width=4,
        tile_height=4,
        overlap=0,
        tile_batch_size=2,
    )
    calls: list[dict[str, Any]] = []

    def apply_model(
        x: torch.Tensor,
        timestep: torch.Tensor,
        **c: object,
    ) -> torch.Tensor:
        """Record tiled model calls and return deterministic output."""

        calls.append({"x": x, "timestep": timestep, "c": c})
        return x + timestep.reshape((-1, 1, 1, 1))

    x = torch.zeros((2, 1, 4, 8), dtype=torch.float32)
    timestep = torch.tensor([0.5, 0.75], dtype=torch.float32)
    c_concat = torch.arange(64, dtype=torch.float32).reshape((2, 1, 4, 8))
    args = {
        "input": x,
        "timestep": timestep,
        "cond_or_uncond": [0, 1],
        "c": {
            "c_crossattn": torch.ones((2, 3, 1), dtype=torch.float32),
            "c_concat": c_concat,
            "transformer_options": {
                "cond_or_uncond": [0, 1],
                "uuids": ["positive", "negative"],
                "sigmas": timestep,
                "sample_sigmas": torch.tensor([1.0, 0.0]),
            },
        },
    }

    output = wrapped_model.wrapper(apply_model, args)

    assert output.shape == x.shape
    assert torch.equal(
        output[:, :, :, :4],
        x[:, :, :, :4] + timestep.reshape(2, 1, 1, 1),
    )
    assert torch.equal(
        output[:, :, :, 4:],
        x[:, :, :, 4:] + timestep.reshape(2, 1, 1, 1),
    )
    assert len(calls) == 1
    assert calls[0]["x"].shape == (4, 1, 4, 4)
    assert torch.equal(calls[0]["timestep"], torch.tensor([0.5, 0.75, 0.5, 0.75]))
    assert calls[0]["c"]["c_crossattn"].shape == (4, 3, 1)
    assert calls[0]["c"]["c_concat"].shape == (4, 1, 4, 4)
    tiled_options = calls[0]["c"]["transformer_options"]
    assert tiled_options["cond_or_uncond"] == [0, 1, 0, 1]
    assert tiled_options["uuids"] == [
        "positive",
        "negative",
        "positive",
        "negative",
    ]
    assert torch.equal(tiled_options["sigmas"], calls[0]["timestep"])
    assert torch.equal(tiled_options["sample_sigmas"], torch.tensor([1.0, 0.0]))


def test_model_wrapper_tiles_singleton_depth_5d_latents() -> None:
    """The wrapper tiles Anima-style BCDHW latents across spatial axes only."""

    model = FakeModel()
    wrapped_model, _plan = multidiffusion_sampling.clone_model_with_multidiffusion(
        model,
        latent_width=8,
        latent_height=4,
        tile_width=4,
        tile_height=4,
        overlap=0,
        tile_batch_size=2,
    )
    calls: list[dict[str, Any]] = []

    def apply_model(
        x: torch.Tensor,
        timestep: torch.Tensor,
        **c: object,
    ) -> torch.Tensor:
        """Record tiled model calls and return deterministic output."""

        calls.append({"x": x, "timestep": timestep, "c": c})
        return x + timestep.reshape((-1, 1, 1, 1, 1))

    x = torch.zeros((1, 16, 1, 4, 8), dtype=torch.float32)
    timestep = torch.tensor([0.5], dtype=torch.float32)
    c_concat = torch.arange(512, dtype=torch.float32).reshape((1, 16, 1, 4, 8))

    output = wrapped_model.wrapper(
        apply_model,
        {
            "input": x,
            "timestep": timestep,
            "cond_or_uncond": [0],
            "c": {"c_concat": c_concat},
        },
    )

    assert output.shape == x.shape
    assert torch.allclose(output, torch.ones_like(x) * 0.5)
    assert len(calls) == 1
    assert calls[0]["x"].shape == (2, 16, 1, 4, 4)
    assert calls[0]["c"]["c_concat"].shape == (2, 16, 1, 4, 4)
    assert torch.equal(calls[0]["timestep"], torch.tensor([0.5, 0.5]))


def test_model_wrapper_averages_overlapping_tiles_uniformly() -> None:
    """Overlapping tile outputs are averaged by coverage count."""

    model = FakeModel()
    wrapped_model, _plan = multidiffusion_sampling.clone_model_with_multidiffusion(
        model,
        latent_width=6,
        latent_height=4,
        tile_width=4,
        tile_height=4,
        overlap=2,
        tile_batch_size=1,
    )
    call_count = 0

    def apply_model(
        x: torch.Tensor,
        timestep: torch.Tensor,
        **c: object,
    ) -> torch.Tensor:
        """Return a distinct constant per tile call."""

        del timestep, c
        nonlocal call_count
        call_count += 1
        return torch.ones_like(x) * float(call_count * 2 - 1)

    x = torch.zeros((1, 1, 4, 6), dtype=torch.float32)
    output = wrapped_model.wrapper(
        apply_model,
        {"input": x, "timestep": torch.tensor([1.0]), "c": {}, "cond_or_uncond": [0]},
    )

    assert call_count == 2
    assert torch.allclose(output[:, :, :, :2], torch.ones((1, 1, 4, 2)))
    assert torch.allclose(output[:, :, :, 2:4], torch.ones((1, 1, 4, 2)) * 2.0)
    assert torch.allclose(output[:, :, :, 4:], torch.ones((1, 1, 4, 2)) * 3.0)


def test_model_wrapper_averages_overlapping_5d_tiles_uniformly() -> None:
    """Overlapping Anima-style tile outputs are averaged by coverage count."""

    model = FakeModel()
    wrapped_model, _plan = multidiffusion_sampling.clone_model_with_multidiffusion(
        model,
        latent_width=6,
        latent_height=4,
        tile_width=4,
        tile_height=4,
        overlap=2,
        tile_batch_size=1,
    )
    call_count = 0

    def apply_model(
        x: torch.Tensor,
        timestep: torch.Tensor,
        **c: object,
    ) -> torch.Tensor:
        """Return a distinct constant per tile call."""

        del timestep, c
        nonlocal call_count
        call_count += 1
        return torch.ones_like(x) * float(call_count * 2 - 1)

    x = torch.zeros((1, 16, 1, 4, 6), dtype=torch.float32)
    output = wrapped_model.wrapper(
        apply_model,
        {"input": x, "timestep": torch.tensor([1.0]), "c": {}, "cond_or_uncond": [0]},
    )

    assert call_count == 2
    assert torch.allclose(output[:, :, :, :, :2], torch.ones((1, 16, 1, 4, 2)))
    assert torch.allclose(
        output[:, :, :, :, 2:4],
        torch.ones((1, 16, 1, 4, 2)) * 2.0,
    )
    assert torch.allclose(output[:, :, :, :, 4:], torch.ones((1, 16, 1, 4, 2)) * 3.0)


def test_model_wrapper_rejects_unsupported_conditioning() -> None:
    """Control conditioning fails closed in the first implementation."""

    model = FakeModel()
    wrapped_model, _plan = multidiffusion_sampling.clone_model_with_multidiffusion(
        model,
        latent_width=8,
        latent_height=4,
        tile_width=4,
        tile_height=4,
        overlap=0,
        tile_batch_size=2,
    )

    with pytest.raises(ValueError, match="regional conditioning or ControlNet"):
        wrapped_model.wrapper(
            lambda x, timestep, **c: x,
            {
                "input": torch.zeros((1, 1, 4, 8)),
                "timestep": torch.tensor([1.0]),
                "c": {"control": object()},
                "cond_or_uncond": [0],
            },
        )


def test_sample_rejects_unipc_before_sampler_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UniPC sampler names fail before ComfyUI sampler lookup."""

    def fail_resolve_sampler(_sampler_name: str) -> FakeSampler:
        """Fail if sampler resolution is reached."""

        raise AssertionError("UniPC rejection should happen before sampler resolution")

    monkeypatch.setattr(sampling_samplers, "resolve_sampler", fail_resolve_sampler)

    with pytest.raises(ValueError, match="not compatible with UniPC"):
        multidiffusion_sampling.sample_multidiffusion(
            model=FakeModel(),
            seed=1,
            steps=1,
            cfg=1.0,
            sampler_name="uni_pc",
            scheduler="normal",
            positive=[],
            negative=[],
            latent_image={"samples": torch.zeros((1, 4, 4, 4))},
            denoise=1.0,
            latent_tile_width=4,
            latent_tile_height=4,
            latent_tile_overlap=0,
            latent_tile_batch_size=1,
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

    def fake_calculate_sigmas(**kwargs: object) -> torch.Tensor:
        """Record scheduler calculation."""

        calls["calculate_sigmas"] = kwargs
        return fixed_sigmas

    monkeypatch.setattr(
        sampling_schedulers,
        "calculate_sigmas",
        fake_calculate_sigmas,
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

    output = multidiffusion_sampling.sample_multidiffusion(
        model=model,
        seed=123,
        steps=2,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        positive=[{"model_conds": {}}],
        negative=[{"model_conds": {}}],
        latent_image=latent_image,
        denoise=1.0,
        latent_tile_width=4,
        latent_tile_height=4,
        latent_tile_overlap=0,
        latent_tile_batch_size=2,
    )

    assert output is not latent_image
    assert output["samples"] is sampled
    assert output["kept"] == "value"
    assert "downscale_ratio_spacial" not in output
    assert calls["sample_custom"]["model"] is not model
    assert calls["sample_custom"]["model"].wrapper is not None
    assert calls["sample_custom"]["sampler"] is sampler
    assert calls["sample_custom"]["noise"] is fixed_noise
    assert calls["sample_custom"]["disable_pbar"] is True


def test_sample_accepts_singleton_depth_5d_latent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anima-style singleton-depth latents pass runtime validation."""

    calls: dict[str, Any] = {}
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
        """Record sample_custom arguments and return a 5D latent."""

        del noise, cfg, received_sampler, sigmas, positive, negative, noise_mask
        del callback, disable_pbar, seed
        calls["model"] = received_model
        calls["latent_image"] = latent_image
        return latent_image + 1.0

    monkeypatch.setattr(comfy_sample, "sample_custom", fake_sample_custom)

    output = multidiffusion_sampling.sample_multidiffusion(
        model=model,
        seed=123,
        steps=2,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        positive=[{"model_conds": {}}],
        negative=[{"model_conds": {}}],
        latent_image={"samples": latent_samples},
        denoise=1.0,
        latent_tile_width=4,
        latent_tile_height=4,
        latent_tile_overlap=0,
        latent_tile_batch_size=2,
    )

    assert calls["model"].wrapper is not None
    assert calls["latent_image"] is latent_samples
    assert torch.equal(output["samples"], latent_samples + 1.0)


def test_sample_rejects_5d_latent_with_non_singleton_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-singleton 5D latents remain unsupported until validated explicitly."""

    monkeypatch.setattr(
        sampling_samplers,
        "resolve_sampler",
        lambda _sampler_name: FakeSampler(),
    )
    monkeypatch.setattr(
        sampling_schedulers,
        "calculate_sigmas",
        lambda **_kwargs: torch.tensor([1.0, 0.0], dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="singleton third axis"):
        multidiffusion_sampling.sample_multidiffusion(
            model=FakeModel(),
            seed=1,
            steps=1,
            cfg=1.0,
            sampler_name="euler",
            scheduler="normal",
            positive=[],
            negative=[],
            latent_image={"samples": torch.zeros((1, 16, 2, 4, 4))},
            denoise=1.0,
            latent_tile_width=4,
            latent_tile_height=4,
            latent_tile_overlap=0,
            latent_tile_batch_size=1,
        )


def test_sample_rejects_unsupported_conditioning() -> None:
    """Regional and ControlNet conditioning fail closed in the first slice."""

    with pytest.raises(ValueError, match="regional conditioning or ControlNet"):
        multidiffusion_sampling.sample_multidiffusion(
            model=FakeModel(),
            seed=1,
            steps=1,
            cfg=1.0,
            sampler_name="euler",
            scheduler="normal",
            positive=[{"area": (4, 4, 0, 0)}],
            negative=[],
            latent_image={"samples": torch.zeros((1, 4, 4, 4))},
            denoise=1.0,
            latent_tile_width=4,
            latent_tile_height=4,
            latent_tile_overlap=0,
            latent_tile_batch_size=1,
        )
