# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Detect standard image-UNet regional-attention capabilities from live graphs."""

from __future__ import annotations

import comfy.latent_formats
import comfy.model_base
import comfy.model_patcher
from comfy.ldm.modules.attention import (
    BasicTransformerBlock,
    SpatialTransformer,
    SpatialVideoTransformer,
)
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel, VideoResBlock

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


class StandardUnetModelCapabilityDetector:
    """Own topology-driven admission to the standard image-UNet backend."""

    def detect(
        self,
        patcher: comfy.model_patcher.ModelPatcher,
    ) -> RegionalModelCapabilities | None:
        """Return capabilities for a compatible two-dimensional UNet graph."""

        base_model = patcher.model
        diffusion_model = getattr(base_model, "diffusion_model", None)
        if not isinstance(base_model, comfy.model_base.BaseModel) or not isinstance(
            diffusion_model, UNetModel
        ):
            return None
        self._validate_image_latent(getattr(base_model, "latent_format", None))
        self._validate_attention_graph(diffusion_model)
        return _STANDARD_UNET_CAPABILITIES

    @staticmethod
    def _validate_image_latent(latent_format: object) -> None:
        """Require a non-temporal two-dimensional Comfy image latent."""

        if (
            not isinstance(latent_format, comfy.latent_formats.LatentFormat)
            or getattr(latent_format, "latent_dimensions", None) != 2
            or getattr(latent_format, "temporal_downscale_ratio", None) != 1
        ):
            raise ValueError(
                "Standard regional attention requires a two-dimensional image "
                "latent with temporal downscale ratio 1."
            )

    @classmethod
    def _validate_attention_graph(cls, diffusion_model: UNetModel) -> None:
        """Require separate image-self and context-cross attention roles."""

        modules = tuple(diffusion_model.modules())
        if any(isinstance(module, SpatialVideoTransformer) for module in modules):
            raise ValueError(
                "Standard regional attention does not support "
                "SpatialVideoTransformer topology."
            )
        if any(isinstance(module, VideoResBlock) for module in modules):
            raise ValueError(
                "Standard regional attention does not support VideoResBlock topology."
            )
        spatial_transformers = tuple(
            module for module in modules if type(module) is SpatialTransformer
        )
        if not spatial_transformers:
            raise ValueError(
                "Standard regional attention requires at least one exact "
                "SpatialTransformer."
            )
        for spatial_transformer in spatial_transformers:
            cls._validate_transformer_blocks(spatial_transformer)

    @staticmethod
    def _validate_transformer_blocks(
        spatial_transformer: SpatialTransformer,
    ) -> None:
        """Validate every block's image and context attention ownership."""

        blocks = tuple(spatial_transformer.transformer_blocks)
        if not blocks or any(
            type(block) is not BasicTransformerBlock for block in blocks
        ):
            raise ValueError(
                "Standard regional attention requires exact BasicTransformerBlock "
                "attention roles."
            )
        for block in blocks:
            if block.disable_self_attn or block.attn1 is None:
                raise ValueError(
                    "Standard regional attention requires attn1 self-attention."
                )
            if block.switch_temporal_ca_to_sa or block.attn2 is None:
                raise ValueError(
                    "Standard regional attention requires attn2 cross-attention."
                )


_STANDARD_UNET_CAPABILITIES = RegionalModelCapabilities(
    model_family=RegionalModelFamily.STANDARD_UNET,
    attention_backend=RegionalAttentionBackend.UNET_ATTN2_PATCH,
    attention_topology=RegionalAttentionTopology.SEPARATE_IMAGE_AND_CONTEXT,
    latent_layout=RegionalLatentLayout.STANDARD_IMAGE_BCHW,
    spatial_patch_support=RegionalSpatialPatchSupport.FULL_AND_SPATIAL_VIEWS,
    control_gligen_policy=RegionalControlGligenPolicy.REJECT,
    reference_latent_policy=RegionalReferenceLatentPolicy.REJECT,
    known_patch_conflicts=(
        RegionalPatchConflict.ATTN2_INPUT_PATCH,
        RegionalPatchConflict.ATTN2_OUTPUT_PATCH,
    ),
)
