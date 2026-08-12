# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the KSampler Contextual Diffusion node contract."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.segs import NativeSegs
from simple_syrup.nodes_v3.ksampler_contextual_diffusion import (
    KSamplerContextualDiffusionV3,
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

    schema = KSamplerContextualDiffusionV3.define_schema()
    required = {value.id: value for value in schema.inputs if not value.optional}
    optional = {value.id: value for value in schema.inputs if value.optional}

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
    assert required["steps"].default == 4
    assert required["cfg"].default == 1.0
    assert required["diffusion_mode"].options == [
        "multidiffusion",
        "mixture_of_diffusers",
    ]
    assert required["diffusion_mode"].default == "multidiffusion"
    assert required["latent_context_size"].default == 96
    assert required["latent_context_overlap"].default == 32
    assert required["latent_context_batch_size"].default == 4
    assert required["global_weight"].default == 1.0
    assert required["global_steps"].default == 1
    assert required["global_decay"].default == 0.5
    assert optional["segs"].io_type == "SEGS"
    assert optional["region_masks"].io_type == "MASK"
    assert optional["regional_prompt_weight"].default == 0.5
    assert optional["region_mask_feather"].default == 0


def test_node_metadata_matches_separate_sampler_contract() -> None:
    """Contextual Diffusion exposes its latent and actual local contexts."""

    schema = KSamplerContextualDiffusionV3.define_schema()

    assert schema.node_id == "SimpleSyrup.KSamplerContextualDiffusion"
    assert schema.display_name == "KSampler (Contextual Diffusion)"
    assert schema.category == "SimpleSyrup/Sampling"
    assert len(schema.outputs) == 2
    assert "composition" in schema.description


def test_v3_schema_names_both_contextual_diffusion_outputs() -> None:
    """The exported v3 schema exposes the workflow-facing context socket."""

    schema = KSamplerContextualDiffusionV3.define_schema()

    assert [output.id for output in schema.outputs] == ["latent", "contexts_segs"]


def test_sample_delegates_every_control_to_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node API owns no contextual planning or runtime behavior."""

    fake_service = _FakeContextualDiffusionService()
    monkeypatch.setattr(
        KSamplerContextualDiffusionV3,
        "service_class",
        staticmethod(lambda: fake_service),
    )
    latent = {"samples": torch.zeros((1, 4, 32, 48))}
    segs = object()

    result, contexts = KSamplerContextualDiffusionV3.execute(
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
    assert contexts is fake_service.contexts
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
            "region_masks": None,
            "regional_prompt_weight": 0.5,
            "region_mask_feather": 0,
        }
    ]


class _FakeContextualDiffusionService:
    """Record node delegation without entering the Comfy runtime."""

    def __init__(self) -> None:
        """Create a stable output and empty call history."""

        self.output: dict[str, Any] = {"samples": torch.ones((1, 4, 32, 48))}
        self.contexts: NativeSegs = ((256, 384), ())
        self.calls: list[dict[str, Any]] = []

    def sample(self, **kwargs: Any) -> Any:
        """Record one call and return the stable latent."""

        self.calls.append(kwargs)
        return type(
            "Result",
            (),
            {"latent": self.output, "contexts": self.contexts},
        )()
