# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Detect the exact installed Anima regional-attention capability surface."""

from __future__ import annotations

import comfy.latent_formats
import comfy.model_base
import comfy.model_patcher
from comfy.ldm.anima.model import Anima as AnimaDiffusionModel

from ..domain.regional_model_capabilities import (
    RegionalAttentionBackend,
    RegionalAttentionTopology,
    RegionalControlGligenPolicy,
    RegionalLatentLayout,
    RegionalModelCapabilities,
    RegionalModelFamily,
    RegionalPatchConflict,
    RegionalReferenceLatentPolicy,
    RegionalSpatialPatchSupport,
)


class AnimaModelCapabilityDetector:
    """Own defensive routing to the specialized Anima attention backend."""

    def detect(
        self,
        patcher: comfy.model_patcher.ModelPatcher,
    ) -> RegionalModelCapabilities | None:
        """Return capabilities only for the proven installed Anima surface."""

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


_ANIMA_CAPABILITIES = RegionalModelCapabilities(
    model_family=RegionalModelFamily.ANIMA,
    attention_backend=RegionalAttentionBackend.ANIMA_OBJECT_PATCH,
    attention_topology=RegionalAttentionTopology.SINGLETON_FRAME_SPATIOTEMPORAL,
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
