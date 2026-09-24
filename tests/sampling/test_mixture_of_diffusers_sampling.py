# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Mixture of Diffusers ComfyUI sampling runtime."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from simple_syrup.domain.segs import CropRegion
from simple_syrup.domain.tiled_diffusion import gaussian_tile_weights
from simple_syrup.runtime import mixture_of_diffusers_sampling as mod_sampling
from simple_syrup.runtime.detail_previews import DetailPreviewContext

comfy_sample = mod_sampling._comfy_sample()
comfy_utils = mod_sampling._comfy_utils()
latent_preview = mod_sampling._latent_preview()


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
    """KSampler-style Mixture of Diffusers keeps generic latent previews."""

    monkeypatch.setattr(
        latent_preview,
        "prepare_callback",
        lambda _model, _steps: "generic callback",
    )

    assert mod_sampling._sampling_callback(FakeModel(), 4, None) == "generic callback"


def test_sampling_callback_uses_detail_preview_with_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detailer Mixture of Diffusers uses the shared detail preview callback."""

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
        mod_sampling,
        "prepare_detail_preview_callback",
        fake_prepare_detail_preview_callback,
    )

    model = FakeModel()
    assert mod_sampling._sampling_callback(model, 4, context) == "detail callback"
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
        self.wrapper: Any = None
        self.model_sampling = object()
        self.parent = parent
        self.clone_count = 0

    def clone(self) -> FakeModel:
        """Return a cloned model with copied options."""

        self.clone_count += 1
        return FakeModel(self.model_options.copy(), parent=self)

    def set_model_unet_function_wrapper(self, wrapper: object) -> None:
        """Capture the installed model function wrapper."""

        self.wrapper = wrapper
        self.model_options["model_function_wrapper"] = wrapper

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


def test_clone_model_with_mixture_composes_differential_on_same_clone() -> None:
    """Differential diffusion is installed without cloning a temporary parent."""

    model = FakeModel()

    wrapped_model, _plan = mod_sampling.clone_model_with_mixture_of_diffusers(
        model,
        latent_width=8,
        latent_height=4,
        tile_width=4,
        tile_height=4,
        overlap=0,
        tile_batch_size=2,
        differential_diffusion=True,
    )

    assert model.clone_count == 1
    assert wrapped_model.parent is model
    assert callable(wrapped_model.model_options["denoise_mask_function"])
    assert isinstance(
        wrapped_model.wrapper,
        mod_sampling.MixtureOfDiffusersModelWrapper,
    )


def test_model_wrapper_tiles_input_conditioning_and_transformer_options() -> None:
    """The wrapper tiles latents, conditioning tensors, timesteps, and metadata."""

    model = FakeModel()
    wrapped_model, plan = mod_sampling.clone_model_with_mixture_of_diffusers(
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
        output[:, :, :, :4], x[:, :, :, :4] + timestep.reshape(2, 1, 1, 1)
    )
    assert torch.equal(
        output[:, :, :, 4:], x[:, :, :, 4:] + timestep.reshape(2, 1, 1, 1)
    )
    assert plan.tile_batch_size == 2
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
    wrapped_model, _plan = mod_sampling.clone_model_with_mixture_of_diffusers(
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


def test_model_wrapper_blends_overlapping_tiles_with_gaussian_weights() -> None:
    """Overlapping tile outputs are blended with Mixture Gaussian weights."""

    model = FakeModel()
    wrapped_model, _plan = mod_sampling.clone_model_with_mixture_of_diffusers(
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
    weights = gaussian_tile_weights(
        4,
        4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_overlap = (1.0 * weights[0, 2] + 3.0 * weights[0, 0]) / (
        weights[0, 2] + weights[0, 0]
    )

    assert call_count == 2
    assert torch.allclose(output[:, :, :, :2], torch.ones((1, 1, 4, 2)))
    assert torch.allclose(output[:, :, :, 4:], torch.ones((1, 1, 4, 2)) * 3.0)
    assert torch.isclose(output[0, 0, 0, 2], expected_overlap)


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
    wrapped_model, _plan = mod_sampling.clone_model_with_mixture_of_diffusers(
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
    """Unexpected model input shapes are delegated without tiling."""

    model = FakeModel()
    wrapped_model, _plan = mod_sampling.clone_model_with_mixture_of_diffusers(
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
