# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact specialized Anima capability detection remains unchanged."""

from __future__ import annotations

from typing import cast

import comfy.latent_formats
import comfy.model_base
import pytest
import torch
from comfy.ldm.anima.model import Anima as AnimaDiffusionModel
from comfy.ldm.cosmos.predict2 import MiniTrainDIT
from regional_model_capability_test_values import empty_module, patcher

from simple_syrup.domain.regional_model_capabilities import (
    RegionalAttentionBackend,
    RegionalAttentionTopology,
    RegionalLatentLayout,
    RegionalModelFamily,
)
from simple_syrup.runtime.anima_model_capability import AnimaModelCapabilityDetector


def test_exact_anima_surface_reports_specialized_capabilities() -> None:
    """Admit the installed Anima wrapper, diffusion module, and latent format."""

    model = patcher(
        base_type=comfy.model_base.Anima,
        diffusion_model=empty_module(AnimaDiffusionModel),
        latent_format=comfy.latent_formats.Wan21(),
    )
    original_options = model.model_options.copy()
    original_object_patches = model.object_patches.copy()

    capabilities = AnimaModelCapabilityDetector().detect(model)

    assert capabilities is not None
    assert capabilities.model_family is RegionalModelFamily.ANIMA
    assert capabilities.attention_backend is RegionalAttentionBackend.ANIMA_OBJECT_PATCH
    assert capabilities.attention_topology is (
        RegionalAttentionTopology.SINGLETON_FRAME_SPATIOTEMPORAL
    )
    assert capabilities.latent_layout is RegionalLatentLayout.ANIMA_SINGLE_FRAME_BCTHW
    assert model.model_options == original_options
    assert model.object_patches == original_object_patches


@pytest.mark.parametrize(
    ("base_type", "diffusion_type", "latent_format"),
    [
        (
            comfy.model_base.CosmosPredict2,
            MiniTrainDIT,
            comfy.latent_formats.Wan21(),
        ),
        (
            comfy.model_base.CosmosPredict2,
            AnimaDiffusionModel,
            comfy.latent_formats.Wan21(),
        ),
        (
            comfy.model_base.Anima,
            MiniTrainDIT,
            comfy.latent_formats.Wan21(),
        ),
        (
            comfy.model_base.Anima,
            AnimaDiffusionModel,
            comfy.latent_formats.SD15(),
        ),
    ],
)
def test_detector_rejects_every_cross_composed_anima_like_surface(
    base_type: type[torch.nn.Module],
    diffusion_type: type[torch.nn.Module],
    latent_format: object,
) -> None:
    """Refuse shared ancestry and cross-composed Anima-like object surfaces."""

    model = patcher(
        base_type=base_type,
        diffusion_model=empty_module(diffusion_type),
        latent_format=latent_format,
    )

    assert AnimaModelCapabilityDetector().detect(model) is None


def test_detector_rejects_anima_wrapper_subclasses() -> None:
    """Keep specialized Anima routing bound to its proven installed surface."""

    anima_subclass = cast(
        type[torch.nn.Module],
        type("AnimaSubclass", (comfy.model_base.Anima,), {}),
    )
    model = patcher(
        base_type=anima_subclass,
        diffusion_model=empty_module(AnimaDiffusionModel),
        latent_format=comfy.latent_formats.Wan21(),
    )

    assert AnimaModelCapabilityDetector().detect(model) is None
