# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the KSampler Contextual Diffusion node contract."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.nodes.ksampler_contextual_diffusion import (
    KSamplerContextualDiffusion,
)
from simple_syrup.runtime import sampling_samplers, sampling_schedulers


def test_input_types_expose_concise_klein_oriented_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node keeps KSampler inputs and bounded contextual settings."""

    monkeypatch.setattr(sampling_samplers, "available_samplers", lambda: ("euler",))
    monkeypatch.setattr(
        sampling_schedulers,
        "available_schedulers",
        lambda: ("simple",),
    )

    declared = KSamplerContextualDiffusion.INPUT_TYPES()
    required = declared["required"]

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
        "diffusion_mode",
        "latent_context_size",
        "latent_context_overlap",
        "latent_context_batch_size",
        "global_weight",
        "global_steps",
        "global_decay",
    )
    assert required["steps"][1]["default"] == 4
    assert required["cfg"][1]["default"] == 1.0
    assert required["diffusion_mode"][0] == [
        "multidiffusion",
        "mixture_of_diffusers",
    ]
    assert required["diffusion_mode"][1]["default"] == "multidiffusion"
    assert required["latent_context_size"][1]["default"] == 96
    assert required["latent_context_overlap"][1]["default"] == 32
    assert required["latent_context_batch_size"][1]["default"] == 4
    assert required["global_weight"][1]["default"] == 1.0
    assert required["global_steps"][1]["default"] == 1
    assert required["global_decay"][1]["default"] == 0.5
    assert declared["optional"]["segs"][0] == "SEGS"


def test_node_metadata_matches_separate_sampler_contract() -> None:
    """Contextual Diffusion remains a distinct sampler with latent output."""

    assert KSamplerContextualDiffusion.RETURN_TYPES == ("LATENT",)
    assert KSamplerContextualDiffusion.FUNCTION == "sample"
    assert KSamplerContextualDiffusion.CATEGORY == "SimpleSyrup/Sampling"
    assert "composition" in KSamplerContextualDiffusion.DESCRIPTION


def test_sample_delegates_every_control_to_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node API owns no contextual planning or runtime behavior."""

    fake_service = _FakeContextualDiffusionService()
    monkeypatch.setattr(
        KSamplerContextualDiffusion,
        "service_class",
        staticmethod(lambda: fake_service),
    )
    latent = {"samples": torch.zeros((1, 4, 32, 48))}
    segs = object()

    (result,) = KSamplerContextualDiffusion().sample(
        model="model",
        seed=12,
        steps=8,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive="positive",
        negative="negative",
        latent_image=latent,
        denoise=0.7,
        diffusion_mode="mixture_of_diffusers",
        latent_context_size=96,
        latent_context_overlap=12,
        latent_context_batch_size=3,
        global_weight=0.9,
        global_steps=2,
        global_decay=0.4,
        segs=segs,
    )

    assert result is fake_service.output
    assert fake_service.calls == [
        {
            "model": "model",
            "seed": 12,
            "steps": 8,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": "positive",
            "negative": "negative",
            "latent_image": latent,
            "denoise": 0.7,
            "diffusion_mode": "mixture_of_diffusers",
            "latent_context_size": 96,
            "latent_context_overlap": 12,
            "latent_context_batch_size": 3,
            "global_weight": 0.9,
            "global_steps": 2,
            "global_decay": 0.4,
            "segs": segs,
        }
    ]


class _FakeContextualDiffusionService:
    """Record node delegation without entering the Comfy runtime."""

    def __init__(self) -> None:
        """Create a stable output and empty call history."""

        self.output: dict[str, Any] = {"samples": torch.ones((1, 4, 32, 48))}
        self.calls: list[dict[str, Any]] = []

    def sample(self, **kwargs: Any) -> dict[str, Any]:
        """Record one call and return the stable latent."""

        self.calls.append(kwargs)
        return self.output
