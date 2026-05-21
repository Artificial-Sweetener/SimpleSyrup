# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the KSampler Extras ComfyUI node."""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Any

import torch

from simple_syrup.nodes.ksampler_extras import KSamplerExtras
from simple_syrup.runtime import sampling_samplers, sampling_schedulers

comfy_sample = import_module("comfy.sample")
comfy_utils = import_module("comfy.utils")
latent_preview = import_module("latent_preview")


class FakeModel:
    """Provide model attributes used by KSampler Extras execution."""

    def __init__(self) -> None:
        """Create a fake sampling model."""

        self.load_device = torch.device("cpu")
        self.model_options: dict[str, object] = {}
        self.model_sampling = object()

    def get_model_object(self, name: str) -> object:
        """Return the requested fake model object."""

        assert name == "model_sampling"
        return self.model_sampling


class FakeSampler:
    """Represent a resolved sampler object in node execution tests."""

    def sample(self, *args: object, **kwargs: object) -> object:
        """Provide the sampler protocol expected by runtime code."""

        return None


def test_input_types_match_simple_ksampler_contract() -> None:
    """The node exposes the same inputs as ComfyUI's simple KSampler."""

    required = KSamplerExtras.INPUT_TYPES()["required"]

    assert tuple(required) == (
        "model",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "denoise",
    )


def test_node_metadata_matches_contract() -> None:
    """The node declares the expected ComfyUI output contract."""

    assert KSamplerExtras.RETURN_TYPES == ("LATENT",)
    assert KSamplerExtras.FUNCTION == "sample"
    assert KSamplerExtras.CATEGORY == "SimpleSyrup/Sampling"


def test_user_facing_text_describes_behavior_without_ownership_language() -> None:
    """User-facing copy describes controls without implementation ownership terms."""

    required = KSamplerExtras.INPUT_TYPES()["required"]
    user_facing_text = [
        KSamplerExtras.DESCRIPTION,
        required["sampler_name"][1]["tooltip"],
        required["scheduler"][1]["tooltip"],
    ]

    assert "algorithm" in required["sampler_name"][1]["tooltip"].lower()
    assert "noise" in required["scheduler"][1]["tooltip"].lower()
    assert all("owned" not in text.lower() for text in user_facing_text)
    assert all("ownership" not in text.lower() for text in user_facing_text)


def test_sampler_options_include_lcm() -> None:
    """Core ComfyUI samplers and local extras are exposed."""

    sampler_options = KSamplerExtras.INPUT_TYPES()["required"]["sampler_name"][0]

    assert "lcm" in sampler_options
    assert "euler_a_a1111" in sampler_options


def test_scheduler_options_include_extras_and_exclude_svd() -> None:
    """The node exposes supported extra schedulers and excludes unsupported SVD."""

    scheduler_options = KSamplerExtras.INPUT_TYPES()["required"]["scheduler"][0]

    assert "AYS SD1" in scheduler_options
    assert "AYS SDXL" in scheduler_options
    assert "GITS" in scheduler_options
    assert "beta57" in scheduler_options
    assert "automatic_a1111" in scheduler_options
    assert "AYS SVD" not in scheduler_options


def test_node_import_does_not_require_efficiency_nodes() -> None:
    """The node does not import Efficiency Nodes as a runtime dependency."""

    assert "efficiency_nodes" not in sys.modules


def test_sample_delegates_to_runtime_helpers(
    monkeypatch: Any,
) -> None:
    """Sampling uses SimpleSyrup runtime helpers and ComfyUI sample_custom."""

    calls: dict[str, Any] = {}
    model = FakeModel()
    sampler = FakeSampler()
    latent_samples = torch.ones((1, 4, 8, 8), dtype=torch.float32)
    fixed_noise = torch.full_like(latent_samples, 2.0)
    fixed_sigmas = torch.tensor([1.0, 0.0], dtype=torch.float32)
    sampled = torch.full_like(latent_samples, 3.0)
    latent_image: dict[str, Any] = {
        "samples": latent_samples,
        "batch_index": [0],
        "noise_mask": torch.ones((1, 1, 8, 8), dtype=torch.float32),
        "downscale_ratio_spacial": 2,
        "kept": "value",
    }

    def fake_resolve_sampler(sampler_name: str) -> FakeSampler:
        """Record sampler resolution."""

        calls["sampler_name"] = sampler_name
        return sampler

    def fake_calculate_sigmas(
        model: FakeModel,
        scheduler_name: str,
        sampler_name: str,
        steps: int,
        denoise: float,
    ) -> torch.Tensor:
        """Record scheduler calculation."""

        calls["calculate_sigmas"] = {
            "model": model,
            "scheduler_name": scheduler_name,
            "sampler_name": sampler_name,
            "steps": steps,
            "denoise": denoise,
        }
        return fixed_sigmas

    def fake_fix_empty_latent_channels(
        received_model: FakeModel,
        samples: torch.Tensor,
        downscale_ratio_spacial: int | None,
    ) -> torch.Tensor:
        """Record latent channel normalization."""

        calls["fix_empty_latent_channels"] = {
            "model": received_model,
            "samples": samples,
            "downscale_ratio_spacial": downscale_ratio_spacial,
        }
        return samples

    def fake_prepare_noise(
        samples: torch.Tensor,
        seed: int,
        batch_inds: list[int],
    ) -> torch.Tensor:
        """Record noise preparation."""

        calls["prepare_noise"] = {
            "samples": samples,
            "seed": seed,
            "batch_inds": batch_inds,
        }
        return fixed_noise

    def fake_prepare_callback(received_model: FakeModel, steps: int) -> str:
        """Record callback preparation."""

        calls["prepare_callback"] = {"model": received_model, "steps": steps}
        return "callback"

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
        callback: str,
        disable_pbar: bool,
        seed: int,
    ) -> torch.Tensor:
        """Record custom sampling arguments."""

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

    monkeypatch.setattr(
        sampling_samplers,
        "resolve_sampler",
        fake_resolve_sampler,
    )
    monkeypatch.setattr(
        sampling_schedulers,
        "calculate_sigmas",
        fake_calculate_sigmas,
    )
    monkeypatch.setattr(
        comfy_sample,
        "fix_empty_latent_channels",
        fake_fix_empty_latent_channels,
    )
    monkeypatch.setattr(
        comfy_sample,
        "prepare_noise",
        fake_prepare_noise,
    )
    monkeypatch.setattr(
        latent_preview,
        "prepare_callback",
        fake_prepare_callback,
    )
    monkeypatch.setattr(
        comfy_sample,
        "sample_custom",
        fake_sample_custom,
    )
    monkeypatch.setattr(comfy_utils, "PROGRESS_BAR_ENABLED", False)

    (output,) = KSamplerExtras().sample(
        model=model,
        seed=123,
        steps=2,
        cfg=7.5,
        sampler_name="lcm",
        scheduler="GITS",
        positive="positive",
        negative="negative",
        latent_image=latent_image,
        denoise=0.8,
    )

    assert output is not latent_image
    assert output["samples"] is sampled
    assert output["kept"] == "value"
    assert "downscale_ratio_spacial" not in output
    assert calls["sampler_name"] == "lcm"
    assert calls["calculate_sigmas"] == {
        "model": model,
        "scheduler_name": "GITS",
        "sampler_name": "lcm",
        "steps": 2,
        "denoise": 0.8,
    }
    assert calls["prepare_noise"]["batch_inds"] == [0]
    assert calls["sample_custom"]["noise_mask"] is latent_image["noise_mask"]
    assert calls["sample_custom"]["sampler"] is sampler
    assert calls["sample_custom"]["sigmas"] is fixed_sigmas
    assert calls["sample_custom"]["disable_pbar"] is True
