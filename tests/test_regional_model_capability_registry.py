# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify ordered regional capability routing and unsupported diagnostics."""

from __future__ import annotations

import comfy.latent_formats
import comfy.model_base
import comfy.model_patcher
import pytest
from comfy.ldm.anima.model import Anima as AnimaDiffusionModel
from comfy.ldm.cosmos.predict2 import MiniTrainDIT
from regional_model_capability_test_values import (
    AlternateImageLatent,
    empty_module,
    patcher,
    standard_unet_graph,
)

from simple_syrup.domain.regional_model_capabilities import RegionalModelFamily
from simple_syrup.runtime.regional_model_capabilities import (
    RegionalModelCapabilityRegistry,
)


def test_registry_routes_one_capability_equivalent_standard_unet() -> None:
    """Select standard execution from live graph evidence rather than type pairs."""

    model = patcher(
        base_type=comfy.model_base.BaseModel,
        diffusion_model=standard_unet_graph(),
        latent_format=AlternateImageLatent(),
    )

    capabilities = RegionalModelCapabilityRegistry().capabilities_for(model)

    assert capabilities.model_family is RegionalModelFamily.STANDARD_UNET


def test_registry_routes_the_exact_specialized_anima_surface() -> None:
    """Keep the specialized detector ahead of generalized topology admission."""

    model = patcher(
        base_type=comfy.model_base.Anima,
        diffusion_model=empty_module(AnimaDiffusionModel),
        latent_format=comfy.latent_formats.Wan21(),
    )

    capabilities = RegionalModelCapabilityRegistry().capabilities_for(model)

    assert capabilities.model_family is RegionalModelFamily.ANIMA


def test_unknown_model_diagnostic_reports_every_observed_type() -> None:
    """Name the patcher, base, diffusion, and latent types needed for diagnosis."""

    model = patcher(
        base_type=comfy.model_base.BaseModel,
        diffusion_model=empty_module(MiniTrainDIT),
        latent_format=comfy.latent_formats.Wan21(),
    )

    with pytest.raises(ValueError) as captured:
        RegionalModelCapabilityRegistry().capabilities_for(model)

    message = str(captured.value)
    assert "patcher=comfy.model_patcher.ModelPatcher" in message
    assert "base_model=comfy.model_base.BaseModel" in message
    assert "diffusion_model=comfy.ldm.cosmos.predict2.MiniTrainDIT" in message
    assert "latent_format=comfy.latent_formats.Wan21" in message


def test_registry_rejects_non_patcher_boundary() -> None:
    """Require the public Comfy MODEL wrapper rather than a loose model object."""

    with pytest.raises(TypeError, match="requires a Comfy ModelPatcher"):
        RegionalModelCapabilityRegistry().capabilities_for(object())


def test_registry_does_not_mutate_the_input_patcher() -> None:
    """Keep detection read-only across model options and object patches."""

    model = patcher(
        base_type=comfy.model_base.BaseModel,
        diffusion_model=standard_unet_graph(),
        latent_format=AlternateImageLatent(),
    )
    original_options = model.model_options.copy()
    original_object_patches = model.object_patches.copy()

    RegionalModelCapabilityRegistry().capabilities_for(model)

    assert model.model_options == original_options
    assert model.object_patches == original_object_patches
    assert isinstance(model, comfy.model_patcher.ModelPatcher)
