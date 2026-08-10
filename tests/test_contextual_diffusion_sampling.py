# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for contextual diffusion prediction fusion."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.contextual_diffusion import (
    ContextualDiffusionControls,
    build_contextual_diffusion_plan,
)
from simple_syrup.runtime import (
    contextual_diffusion_sampling,
    sampling_samplers,
    sampling_schedulers,
)
from simple_syrup.runtime.contextual_diffusion_sampling import (
    ContextualDiffusionModelWrapper,
)

comfy_sample = contextual_diffusion_sampling._comfy_sample()
comfy_utils = contextual_diffusion_sampling._comfy_utils()
latent_preview = contextual_diffusion_sampling._latent_preview()


def test_global_prediction_owns_low_frequency_while_tiles_keep_detail() -> None:
    """The native view replaces tile-level scene intent without blurring detail."""

    controls = _controls()
    plan = build_contextual_diffusion_plan(
        latent_width=32,
        latent_height=16,
        controls=controls,
        segs=None,
    )
    wrapper = ContextualDiffusionModelWrapper(
        plan=plan,
        controls=controls,
        sigmas=torch.tensor([1.0, 0.0]),
        existing_wrapper=None,
    )
    calls: list[tuple[int, int]] = []

    def apply_model(
        x: torch.Tensor,
        timestep: torch.Tensor,
        **conditioning: object,
    ) -> torch.Tensor:
        """Return high-frequency tile detail and a constant whole-image intent."""

        del timestep, conditioning
        calls.append((int(x.shape[-2]), int(x.shape[-1])))
        if x.shape[-2:] == (8, 16):
            return torch.full_like(x, 3.0)
        rows = torch.arange(x.shape[-2], device=x.device).reshape(1, 1, -1, 1)
        return torch.where(rows % 2 == 0, 1.0, -1.0).expand_as(x)

    output = wrapper(
        apply_model,
        {
            "input": torch.zeros((1, 1, 16, 32)),
            "timestep": torch.tensor([1.0]),
            "c": {},
        },
    )

    assert calls == [(16, 16), (8, 16)]
    assert torch.allclose(output[:, :, 0::2], torch.full((1, 1, 8, 32), 4.0))
    assert torch.allclose(output[:, :, 1::2], torch.full((1, 1, 8, 32), 2.0))


def test_global_prediction_decays_then_stops_after_configured_steps() -> None:
    """Global authority decays before late denoising becomes local-only."""

    controls = _controls(global_steps=2)
    plan = build_contextual_diffusion_plan(
        latent_width=32,
        latent_height=16,
        controls=controls,
        segs=None,
    )
    wrapper = ContextualDiffusionModelWrapper(
        plan=plan,
        controls=controls,
        sigmas=torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]),
        existing_wrapper=None,
    )
    calls: list[tuple[int, int]] = []

    def apply_model(
        x: torch.Tensor,
        timestep: torch.Tensor,
        **conditioning: object,
    ) -> torch.Tensor:
        """Record whether each prediction uses local or global context."""

        del timestep, conditioning
        calls.append((int(x.shape[-2]), int(x.shape[-1])))
        value = 3.0 if x.shape[-2:] == (8, 16) else 1.0
        return torch.full_like(x, value)

    base_args = {
        "input": torch.zeros((1, 1, 16, 32)),
        "c": {},
    }
    output = wrapper(apply_model, base_args | {"timestep": torch.tensor([0.75])})
    assert calls == [(16, 16), (8, 16)]
    assert torch.allclose(output, torch.full_like(output, 2.0))

    calls.clear()
    output = wrapper(apply_model, base_args | {"timestep": torch.tensor([0.5])})
    assert calls == [(16, 16)]
    assert torch.allclose(output, torch.ones_like(output))


def test_every_prediction_receives_each_complete_reference_image() -> None:
    """Local and global predictions preserve ordered multi-image conditioning."""

    controls = _controls()
    plan = build_contextual_diffusion_plan(
        latent_width=32,
        latent_height=16,
        controls=controls,
        segs=None,
    )
    wrapper = ContextualDiffusionModelWrapper(
        plan=plan,
        controls=controls,
        sigmas=torch.tensor([1.0, 0.0]),
        existing_wrapper=None,
    )
    image_1 = torch.arange(1 * 4 * 16 * 32, dtype=torch.float32).reshape((1, 4, 16, 32))
    image_2 = torch.full((1, 4, 10, 12), 2.0)
    received: list[list[torch.Tensor]] = []

    def apply_model(
        x: torch.Tensor,
        timestep: torch.Tensor,
        **conditioning: object,
    ) -> torch.Tensor:
        """Capture reference batches presented to each bounded prediction."""

        del timestep
        references = conditioning["ref_latents"]
        assert isinstance(references, list)
        assert all(isinstance(reference, torch.Tensor) for reference in references)
        received.append(references)
        return torch.zeros_like(x)

    wrapper(
        apply_model,
        {
            "input": torch.zeros((1, 1, 16, 32)),
            "timestep": torch.tensor([1.0]),
            "c": {
                "ref_latents": [image_1, image_2],
                "ref_latents_method": "index",
            },
        },
    )

    assert len(received) == 2
    assert torch.equal(received[0][0], torch.cat([image_1, image_1], dim=0))
    assert torch.equal(received[0][1], torch.cat([image_2, image_2], dim=0))
    assert torch.equal(received[1][0], image_1)
    assert torch.equal(received[1][1], image_2)


def test_native_sized_canvas_delegates_to_one_original_model_call() -> None:
    """A canvas already inside the model view limit behaves like normal sampling."""

    controls = _controls()
    plan = build_contextual_diffusion_plan(
        latent_width=16,
        latent_height=16,
        controls=controls,
        segs=None,
    )
    wrapper = ContextualDiffusionModelWrapper(
        plan=plan,
        controls=controls,
        sigmas=torch.tensor([1.0, 0.0]),
        existing_wrapper=None,
    )
    calls = 0

    def apply_model(
        x: torch.Tensor,
        timestep: torch.Tensor,
        **conditioning: object,
    ) -> torch.Tensor:
        """Count direct model evaluations."""

        del timestep, conditioning
        nonlocal calls
        calls += 1
        return x + 2.0

    x = torch.zeros((1, 1, 16, 16))
    output = wrapper(
        apply_model,
        {"input": x, "timestep": torch.tensor([1.0]), "c": {}},
    )

    assert calls == 1
    assert torch.equal(output, x + 2.0)


def test_runtime_delegates_sampling_to_comfy_with_wrapped_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vertical runtime path preserves KSampler sampling and latent metadata."""

    model = _FakeModel()
    sampler = object()
    latent_samples = torch.zeros((1, 4, 16, 32))
    sampled = torch.ones_like(latent_samples)
    latent = {
        "samples": latent_samples,
        "downscale_ratio_spacial": 2,
        "kept": "metadata",
    }
    controls = _controls()
    plan = build_contextual_diffusion_plan(
        latent_width=32,
        latent_height=16,
        controls=controls,
        segs=None,
    )
    calls: dict[str, Any] = {}

    monkeypatch.setattr(
        sampling_samplers,
        "resolve_sampler",
        lambda _name: sampler,
    )
    monkeypatch.setattr(
        sampling_schedulers,
        "calculate_sigmas",
        lambda **_kwargs: torch.tensor([1.0, 0.0]),
    )
    monkeypatch.setattr(
        comfy_sample,
        "fix_empty_latent_channels",
        lambda _model, samples, _ratio: samples,
    )
    monkeypatch.setattr(
        comfy_sample,
        "prepare_noise",
        lambda samples, _seed, _batch_inds=None: torch.ones_like(samples),
    )
    monkeypatch.setattr(latent_preview, "prepare_callback", lambda _model, _steps: None)
    monkeypatch.setattr(comfy_utils, "PROGRESS_BAR_ENABLED", False)
    mask_support: list[bool] = []

    def record_conditioning_policy(
        _conditioning: object,
        *,
        sampler_label: str,
        allow_full_context_masks: bool = False,
    ) -> None:
        """Record the regional mask policy at the runtime boundary."""

        assert sampler_label == "Contextual Diffusion"
        mask_support.append(allow_full_context_masks)

    monkeypatch.setattr(
        contextual_diffusion_sampling,
        "reject_unsupported_conditioning",
        record_conditioning_policy,
    )

    def fake_sample_custom(
        sampling_model: _FakeModel,
        noise: torch.Tensor,
        cfg: float,
        received_sampler: object,
        sigmas: torch.Tensor,
        positive: object,
        negative: object,
        latent_image: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Capture the final Comfy boundary call."""

        del noise, cfg, sigmas, positive, negative, latent_image, kwargs
        calls["model"] = sampling_model
        calls["sampler"] = received_sampler
        return sampled

    monkeypatch.setattr(comfy_sample, "sample_custom", fake_sample_custom)

    output = contextual_diffusion_sampling.sample_contextual_diffusion(
        model=model,
        seed=7,
        steps=2,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive=[],
        negative=[],
        latent_image=latent,
        denoise=0.8,
        diffusion_mode="mixture_of_diffusers",
        controls=controls,
        plan=plan,
        allow_full_context_masks=True,
    )

    assert calls["model"] is not model
    assert isinstance(calls["model"].wrapper, ContextualDiffusionModelWrapper)
    assert calls["model"].wrapper.diffusion_mode == "mixture_of_diffusers"
    assert calls["sampler"] is sampler
    assert output["samples"] is sampled
    assert output["kept"] == "metadata"
    assert "downscale_ratio_spacial" not in output
    assert mask_support == [True, True]


def _controls(
    *,
    global_steps: int = 1,
    global_decay: float = 0.5,
) -> ContextualDiffusionControls:
    """Return a small two-tile test configuration."""

    return ContextualDiffusionControls(
        latent_context_size=16,
        latent_context_overlap=0,
        latent_context_batch_size=2,
        global_weight=1.0,
        global_steps=global_steps,
        global_decay=global_decay,
    )


class _FakeModel:
    """Provide the ModelPatcher surface used by the semantic runtime."""

    def __init__(
        self,
        model_options: dict[str, Any] | None = None,
        parent: _FakeModel | None = None,
    ) -> None:
        """Create a CPU-backed fake model patcher."""

        self.load_device = torch.device("cpu")
        self.model_options = {} if model_options is None else model_options
        self.wrapper: object | None = None
        self.parent = parent

    def clone(self) -> _FakeModel:
        """Return a clone with copied model options."""

        return _FakeModel(self.model_options.copy(), parent=self)

    def set_model_unet_function_wrapper(self, wrapper: object) -> None:
        """Capture the installed wrapper."""

        self.wrapper = wrapper
        self.model_options["model_function_wrapper"] = wrapper
