# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the Contextual Diffusion Comfy sampling boundary."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.contextual_diffusion import (
    ContextualDiffusionControls,
    build_contextual_diffusion_plan,
)
from simple_syrup.domain.regional_features import (
    RegionalCapabilityAdmission,
    RegionalFeature,
    RegionalFeatureRequest,
)
from simple_syrup.runtime import (
    contextual_diffusion_sampling,
    sampling_samplers,
    sampling_schedulers,
)
from simple_syrup.runtime.contextual_model_wrapper import (
    ContextualDiffusionModelWrapper,
)

COMFY_SAMPLE = contextual_diffusion_sampling._comfy_sample()
COMFY_UTILS = contextual_diffusion_sampling._comfy_utils()
LATENT_PREVIEW = contextual_diffusion_sampling._latent_preview()


def test_runtime_delegates_sampling_to_comfy_with_wrapped_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve KSampler invocation, mask admission, and latent metadata."""

    model = _FakeModel()
    sampler = object()
    latent_samples = torch.zeros((1, 4, 16, 32))
    sampled = torch.ones_like(latent_samples)
    latent = {
        "samples": latent_samples,
        "downscale_ratio_spacial": 2,
        "kept": "metadata",
    }
    controls = ContextualDiffusionControls(16, 0, 2, 1.0, 1, 0.5)
    plan = build_contextual_diffusion_plan(
        latent_width=32,
        latent_height=16,
        controls=controls,
        segs=None,
    )
    calls: dict[str, Any] = {}

    monkeypatch.setattr(sampling_samplers, "resolve_sampler", lambda _name: sampler)
    monkeypatch.setattr(
        sampling_schedulers,
        "calculate_sigmas",
        lambda **_kwargs: torch.tensor([1.0, 0.0]),
    )
    monkeypatch.setattr(
        COMFY_SAMPLE,
        "fix_empty_latent_channels",
        lambda _model, samples, _ratio: samples,
    )
    monkeypatch.setattr(
        COMFY_SAMPLE,
        "prepare_noise",
        lambda samples, _seed, _batch_inds=None: torch.ones_like(samples),
    )
    monkeypatch.setattr(LATENT_PREVIEW, "prepare_callback", lambda _model, _steps: None)
    monkeypatch.setattr(COMFY_UTILS, "PROGRESS_BAR_ENABLED", False)
    mask_support: list[bool] = []

    def record_conditioning_policy(
        _conditioning: object,
        *,
        sampler_label: str,
        capability_admission: RegionalCapabilityAdmission,
    ) -> None:
        """Record admission at the runtime boundary."""

        assert sampler_label == "Contextual Diffusion"
        mask_support.append(
            capability_admission.supports(
                RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING
            )
        )

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
        """Capture the final Comfy sampling call."""

        del noise, cfg, sigmas, positive, negative, latent_image, kwargs
        calls["model"] = sampling_model
        calls["sampler"] = received_sampler
        return sampled

    monkeypatch.setattr(COMFY_SAMPLE, "sample_custom", fake_sample_custom)

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
        capability_admission=_full_context_admission(),
    )

    assert calls["model"] is not model
    assert isinstance(calls["model"].wrapper, ContextualDiffusionModelWrapper)
    assert calls["model"].wrapper.diffusion_mode == "mixture_of_diffusers"
    assert calls["sampler"] is sampler
    assert output["samples"] is sampled
    assert output["kept"] == "metadata"
    assert "downscale_ratio_spacial" not in output
    assert mask_support == [True, True]


def _full_context_admission() -> RegionalCapabilityAdmission:
    """Return one successful full-context mask admission for the runtime test."""

    request = RegionalFeatureRequest(
        frozenset({RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING})
    )
    return RegionalCapabilityAdmission(request, request.features, None)


class _FakeModel:
    """Provide the ModelPatcher surface used by the sampling boundary."""

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
