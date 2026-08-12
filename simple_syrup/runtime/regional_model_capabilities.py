# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Detect regional model capabilities from installed Comfy model objects."""

from __future__ import annotations

from typing import Protocol

import comfy.latent_formats
import comfy.model_base
import comfy.model_patcher
from comfy.ldm.anima.model import Anima as AnimaDiffusionModel
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

from ..domain.regional_model_capabilities import (
    RegionalAttentionBackend,
    RegionalControlGligenPolicy,
    RegionalLatentLayout,
    RegionalModelCapabilities,
    RegionalModelFamily,
    RegionalPatchConflict,
    RegionalReferenceLatentPolicy,
    RegionalSpatialPatchSupport,
)


class _CapabilityDetector(Protocol):
    """Define one read-only exact-type capability detector."""

    def detect(
        self,
        patcher: comfy.model_patcher.ModelPatcher,
    ) -> RegionalModelCapabilities | None:
        """Return capabilities only for the detector's exact admitted family."""


class _AnimaCapabilityDetector:
    """Admit only the installed Anima wrapper, diffusion, and latent format."""

    def detect(
        self,
        patcher: comfy.model_patcher.ModelPatcher,
    ) -> RegionalModelCapabilities | None:
        """Return exact Anima capabilities without classifying Cosmos variants."""

        base_model = patcher.model
        if type(base_model) is not comfy.model_base.Anima:
            return None
        if (
            type(getattr(base_model, "diffusion_model", None))
            is not AnimaDiffusionModel
        ):
            return None
        if (
            type(getattr(base_model, "latent_format", None))
            is not comfy.latent_formats.Wan21
        ):
            return None
        return _ANIMA_CAPABILITIES


class _StandardUNetCapabilityDetector:
    """Admit exact standard SD and SDXL UNet object combinations."""

    _BASE_LATENT_TYPE_PAIRS = (
        (comfy.model_base.BaseModel, comfy.latent_formats.SD15),
        (comfy.model_base.SDXL, comfy.latent_formats.SDXL),
        (comfy.model_base.SDXLRefiner, comfy.latent_formats.SDXL),
    )

    def detect(
        self,
        patcher: comfy.model_patcher.ModelPatcher,
    ) -> RegionalModelCapabilities | None:
        """Return standard UNet capabilities for the explicit exact-type set."""

        base_model = patcher.model
        if type(getattr(base_model, "diffusion_model", None)) is not UNetModel:
            return None
        observed_pair = (
            type(base_model),
            type(getattr(base_model, "latent_format", None)),
        )
        if observed_pair not in self._BASE_LATENT_TYPE_PAIRS:
            return None
        return _STANDARD_UNET_CAPABILITIES


class RegionalModelCapabilityRegistry:
    """Own ordered defensive capability detection for supported model families."""

    def __init__(self) -> None:
        """Install exactly one authoritative detector per supported family."""

        self._detectors: tuple[_CapabilityDetector, ...] = (
            _AnimaCapabilityDetector(),
            _StandardUNetCapabilityDetector(),
        )

    def capabilities_for(self, model: object) -> RegionalModelCapabilities:
        """Return capabilities or report every observed unsupported object type."""

        if not isinstance(model, comfy.model_patcher.ModelPatcher):
            raise TypeError(
                "Regional attention requires a Comfy ModelPatcher; received "
                f"{_type_name(model)}."
            )
        for detector in self._detectors:
            capabilities = detector.detect(model)
            if capabilities is not None:
                return capabilities
        base_model = model.model
        diffusion_model = getattr(base_model, "diffusion_model", None)
        latent_format = getattr(base_model, "latent_format", None)
        raise ValueError(
            "Regional attention does not support this model combination: "
            f"patcher={_type_name(model)}, base_model={_type_name(base_model)}, "
            f"diffusion_model={_type_name(diffusion_model)}, "
            f"latent_format={_type_name(latent_format)}."
        )


def _type_name(value: object) -> str:
    """Return a diagnostic fully qualified runtime type name."""

    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


_ANIMA_CAPABILITIES = RegionalModelCapabilities(
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

_STANDARD_UNET_CAPABILITIES = RegionalModelCapabilities(
    model_family=RegionalModelFamily.STANDARD_UNET,
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
