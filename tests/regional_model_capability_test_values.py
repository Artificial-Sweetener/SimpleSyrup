# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build lightweight installed-Comfy graphs for capability characterization."""

from __future__ import annotations

from typing import Any, TypeVar, cast

import comfy.latent_formats
import comfy.model_base
import comfy.model_patcher
import torch
from comfy.ldm.modules.attention import SpatialTransformer, SpatialVideoTransformer
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel, VideoResBlock

ModuleType = TypeVar("ModuleType", bound=torch.nn.Module)


AlternateImageLatent = cast(
    type[object],
    type(
        "AlternateImageLatent",
        (comfy.latent_formats.LatentFormat,),
        {"latent_dimensions": 2, "temporal_downscale_ratio": 1},
    ),
)
"""Represent one capability-equivalent two-dimensional image latent."""

TemporalLatent = cast(
    type[object],
    type(
        "TemporalLatent",
        (comfy.latent_formats.LatentFormat,),
        {"latent_dimensions": 3, "temporal_downscale_ratio": 4},
    ),
)
"""Represent one unsupported temporal latent without model identity policy."""


def empty_module(module_type: type[ModuleType]) -> ModuleType:
    """Create one exact Torch module instance without allocating model weights."""

    module = module_type.__new__(module_type)
    torch.nn.Module.__init__(module)
    return module


def patcher(
    *,
    base_type: type[torch.nn.Module],
    diffusion_model: torch.nn.Module,
    latent_format: object,
) -> comfy.model_patcher.ModelPatcher:
    """Build a real lightweight ModelPatcher around explicit runtime evidence."""

    base_model = empty_module(base_type)
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


def standard_unet_graph(
    *,
    diffusion_type: type[UNetModel] = UNetModel,
    include_spatial_transformer: bool = True,
    disable_self_attention: bool = False,
    missing_cross_attention: bool = False,
    video_transformer: bool = False,
    video_resblock: bool = False,
) -> UNetModel:
    """Build one minimal UNet graph with a declared attention-role topology."""

    diffusion = empty_module(diffusion_type)
    if include_spatial_transformer:
        if video_transformer:
            spatial: torch.nn.Module = empty_module(SpatialVideoTransformer)
        else:
            spatial_transformer = SpatialTransformer(
                32,
                4,
                8,
                depth=1,
                context_dim=64,
                disable_self_attn=disable_self_attention,
                use_linear=True,
                use_checkpoint=False,
            )
            if missing_cross_attention:
                blocks = cast(Any, spatial_transformer.transformer_blocks)
                blocks[0].attn2 = None
            spatial = spatial_transformer
        diffusion.spatial = spatial
    if video_resblock:
        diffusion.temporal_resblock = empty_module(VideoResBlock)
    return diffusion
