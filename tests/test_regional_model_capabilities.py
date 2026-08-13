# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify defensive regional model capability detection and reporting."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, TypeVar, cast

import comfy.latent_formats
import comfy.model_base
import comfy.model_patcher
import pytest
import torch
from comfy.ldm.anima.model import Anima as AnimaDiffusionModel
from comfy.ldm.cosmos.predict2 import MiniTrainDIT
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

from simple_syrup.domain.regional_model_capabilities import (
    RegionalAttentionBackend,
    RegionalControlGligenPolicy,
    RegionalLatentLayout,
    RegionalModelCapabilities,
    RegionalModelFamily,
    RegionalPatchConflict,
    RegionalReferenceLatentPolicy,
    RegionalSpatialPatchSupport,
)
from simple_syrup.runtime.regional_model_capabilities import (
    RegionalModelCapabilityRegistry,
)

ModuleType = TypeVar("ModuleType", bound=torch.nn.Module)


def test_exact_anima_model_reports_complete_capabilities() -> None:
    """Admit the installed Anima wrapper, diffusion module, and latent format."""

    patcher = _patcher(
        base_type=comfy.model_base.Anima,
        diffusion_type=AnimaDiffusionModel,
        latent_format=comfy.latent_formats.Wan21(),
    )
    original_options = patcher.model_options.copy()
    original_object_patches = patcher.object_patches.copy()

    capabilities = RegionalModelCapabilityRegistry().capabilities_for(patcher)

    assert capabilities == RegionalModelCapabilities(
        model_family=RegionalModelFamily.ANIMA,
        attention_backend=RegionalAttentionBackend.ANIMA_OBJECT_PATCH,
        latent_layout=RegionalLatentLayout.ANIMA_SINGLE_FRAME_BCTHW,
        spatial_patch_support=RegionalSpatialPatchSupport.FULL_AND_SPATIAL_VIEWS,
        control_gligen_policy=RegionalControlGligenPolicy.REJECT,
        reference_latent_policy=RegionalReferenceLatentPolicy.REJECT,
        known_patch_conflicts=(
            RegionalPatchConflict.DIFFUSION_MODEL_WRAPPER,
            RegionalPatchConflict.CROSS_ATTENTION_OBJECT_PATCH,
            RegionalPatchConflict.ATTN2_INPUT_PATCH,
            RegionalPatchConflict.ATTN2_OUTPUT_PATCH,
        ),
    )
    assert patcher.model_options == original_options
    assert patcher.object_patches == original_object_patches


@pytest.mark.parametrize(
    ("base_type", "latent_format"),
    [
        (comfy.model_base.BaseModel, comfy.latent_formats.SD15()),
        (comfy.model_base.SDXL, comfy.latent_formats.SDXL()),
        (comfy.model_base.SDXLRefiner, comfy.latent_formats.SDXL()),
    ],
)
def test_exact_standard_unet_models_report_complete_capabilities(
    base_type: type[torch.nn.Module],
    latent_format: object,
) -> None:
    """Admit only the explicit standard SD and SDXL wrapper/latent set."""

    patcher = _patcher(
        base_type=base_type,
        diffusion_type=UNetModel,
        latent_format=latent_format,
    )
    original_options = patcher.model_options.copy()
    original_object_patches = patcher.object_patches.copy()

    capabilities = RegionalModelCapabilityRegistry().capabilities_for(patcher)

    assert capabilities.model_family is RegionalModelFamily.STANDARD_UNET
    assert capabilities.attention_backend is RegionalAttentionBackend.UNET_ATTN2_PATCH
    assert capabilities.latent_layout is RegionalLatentLayout.STANDARD_IMAGE_BCHW
    assert capabilities.known_patch_conflicts == (
        RegionalPatchConflict.ATTN2_INPUT_PATCH,
        RegionalPatchConflict.ATTN2_OUTPUT_PATCH,
    )
    assert patcher.model_options == original_options
    assert patcher.object_patches == original_object_patches


@pytest.mark.parametrize(
    ("base_type", "latent_format"),
    [
        (comfy.model_base.BaseModel, comfy.latent_formats.SDXL()),
        (comfy.model_base.SDXL, comfy.latent_formats.SD15()),
        (comfy.model_base.SDXLRefiner, comfy.latent_formats.SD15()),
    ],
)
def test_standard_unet_detector_rejects_cross_family_wrapper_latent_pairs(
    base_type: type[torch.nn.Module],
    latent_format: object,
) -> None:
    """Require each admitted wrapper to retain its exact installed latent family."""

    with pytest.raises(ValueError, match="does not support this model combination"):
        RegionalModelCapabilityRegistry().capabilities_for(
            _patcher(
                base_type=base_type,
                diffusion_type=UNetModel,
                latent_format=latent_format,
            )
        )


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
    ids=(
        "cosmos-predict2",
        "cosmos-wrapper-anima-diffusion",
        "anima-wrapper-cosmos-diffusion",
        "anima-wrong-latent",
    ),
)
def test_anima_detector_rejects_every_unverified_cosmos_like_combination(
    base_type: type[torch.nn.Module],
    diffusion_type: type[torch.nn.Module],
    latent_format: object,
) -> None:
    """Refuse shared implementation ancestry and cross-composed Anima-like types."""

    with pytest.raises(ValueError, match="does not support this model combination"):
        RegionalModelCapabilityRegistry().capabilities_for(
            _patcher(
                base_type=base_type,
                diffusion_type=diffusion_type,
                latent_format=latent_format,
            )
        )


def test_detectors_reject_unverified_subclasses() -> None:
    """Require exact installed classes instead of permissive subclass admission."""

    anima_subclass = cast(
        type[torch.nn.Module],
        type("AnimaSubclass", (comfy.model_base.Anima,), {}),
    )
    unet_subclass = cast(
        type[torch.nn.Module],
        type("UNetSubclass", (UNetModel,), {}),
    )

    candidates = (
        _patcher(
            base_type=anima_subclass,
            diffusion_type=AnimaDiffusionModel,
            latent_format=comfy.latent_formats.Wan21(),
        ),
        _patcher(
            base_type=comfy.model_base.BaseModel,
            diffusion_type=unet_subclass,
            latent_format=comfy.latent_formats.SD15(),
        ),
    )

    for patcher in candidates:
        with pytest.raises(ValueError, match="does not support this model combination"):
            RegionalModelCapabilityRegistry().capabilities_for(patcher)


def test_unknown_model_diagnostic_reports_every_observed_type() -> None:
    """Name the patcher, base, diffusion, and latent types needed for diagnosis."""

    patcher = _patcher(
        base_type=comfy.model_base.BaseModel,
        diffusion_type=MiniTrainDIT,
        latent_format=comfy.latent_formats.Wan21(),
    )

    with pytest.raises(ValueError) as captured:
        RegionalModelCapabilityRegistry().capabilities_for(patcher)

    message = str(captured.value)
    assert "patcher=comfy.model_patcher.ModelPatcher" in message
    assert "base_model=comfy.model_base.BaseModel" in message
    assert "diffusion_model=comfy.ldm.cosmos.predict2.MiniTrainDIT" in message
    assert "latent_format=comfy.latent_formats.Wan21" in message


def test_registry_rejects_non_patcher_boundary() -> None:
    """Require the public Comfy MODEL wrapper rather than a loose model object."""

    with pytest.raises(TypeError, match="requires a Comfy ModelPatcher"):
        RegionalModelCapabilityRegistry().capabilities_for(object())


def test_capability_value_is_frozen() -> None:
    """Prevent admission policy from changing after model detection."""

    capabilities = RegionalModelCapabilityRegistry().capabilities_for(
        _patcher(
            base_type=comfy.model_base.Anima,
            diffusion_type=AnimaDiffusionModel,
            latent_format=comfy.latent_formats.Wan21(),
        )
    )
    attribute = "model_family"

    with pytest.raises(FrozenInstanceError):
        setattr(capabilities, attribute, RegionalModelFamily.STANDARD_UNET)


@pytest.mark.parametrize(
    ("conflicts", "error", "message"),
    [
        (cast(Any, [RegionalPatchConflict.ATTN2_INPUT_PATCH]), TypeError, "tuple"),
        ((), ValueError, "require known patch conflicts"),
        (
            cast(Any, ("attn2_input_patch",)),
            TypeError,
            "must contain RegionalPatchConflict",
        ),
        (
            (
                RegionalPatchConflict.ATTN2_INPUT_PATCH,
                RegionalPatchConflict.ATTN2_INPUT_PATCH,
            ),
            ValueError,
            "must be unique and ordered",
        ),
    ],
)
def test_capability_value_rejects_malformed_conflicts(
    conflicts: tuple[RegionalPatchConflict, ...],
    error: type[Exception],
    message: str,
) -> None:
    """Require immutable typed unique patch-conflict reporting."""

    with pytest.raises(error, match=message):
        RegionalModelCapabilities(
            model_family=RegionalModelFamily.STANDARD_UNET,
            attention_backend=RegionalAttentionBackend.UNET_ATTN2_PATCH,
            latent_layout=RegionalLatentLayout.STANDARD_IMAGE_BCHW,
            spatial_patch_support=RegionalSpatialPatchSupport.FULL_AND_SPATIAL_VIEWS,
            control_gligen_policy=RegionalControlGligenPolicy.REJECT,
            reference_latent_policy=RegionalReferenceLatentPolicy.REJECT,
            known_patch_conflicts=conflicts,
        )


def test_capability_value_rejects_cross_family_contract() -> None:
    """Prevent a detector from reporting an incoherent backend/layout combination."""

    with pytest.raises(ValueError, match="do not match the model-family contract"):
        RegionalModelCapabilities(
            model_family=RegionalModelFamily.ANIMA,
            attention_backend=RegionalAttentionBackend.UNET_ATTN2_PATCH,
            latent_layout=RegionalLatentLayout.STANDARD_IMAGE_BCHW,
            spatial_patch_support=RegionalSpatialPatchSupport.FULL_AND_SPATIAL_VIEWS,
            control_gligen_policy=RegionalControlGligenPolicy.REJECT,
            reference_latent_policy=RegionalReferenceLatentPolicy.REJECT,
            known_patch_conflicts=(
                RegionalPatchConflict.ATTN2_INPUT_PATCH,
                RegionalPatchConflict.ATTN2_OUTPUT_PATCH,
            ),
        )


def _patcher(
    *,
    base_type: type[torch.nn.Module],
    diffusion_type: type[torch.nn.Module],
    latent_format: object,
) -> comfy.model_patcher.ModelPatcher:
    """Build a lightweight real ModelPatcher around exact uninitialized types."""

    base_model = _empty_module(base_type)
    diffusion_model = _empty_module(diffusion_type)
    dynamic_base = cast(Any, base_model)
    dynamic_base.diffusion_model = diffusion_model
    dynamic_base.latent_format = latent_format
    dynamic_base.device = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(
        base_model,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        size=1,
    )


def _empty_module(module_type: type[ModuleType]) -> ModuleType:
    """Create one exact torch module instance without allocating model weights."""

    module = module_type.__new__(module_type)
    torch.nn.Module.__init__(module)
    return module
