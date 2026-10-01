# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compile capability configuration into one authoritative sampling execution."""

from __future__ import annotations

from typing import Any, TypedDict

import torch

from ..domain.sampler_options import SamplerOptions, TilingOptions
from ..runtime import sampling_schedulers
from ..runtime.noise_inversion import validate_inversion_target
from ..runtime.tiled_sampling_validation import validate_sampling_controls
from .attention_coupling_sampling_service import AttentionCouplingSamplingService
from .contextual_attention_coupling_sampling_service import (
    ContextualAttentionCouplingSamplingService,
)
from .contextual_diffusion_sampling_service import ContextualDiffusionSamplingService
from .ksampler_sampling_service import KSamplerSamplingService
from .tiled_attention_coupling_sampling_service import (
    TiledAttentionCouplingSamplingService,
)
from .tiled_diffusion_sampling_service import TiledDiffusionSamplingService


class SamplingArguments(TypedDict):
    """Narrow controls while retaining dynamic host MODEL and tensor payloads."""

    model: Any
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    positive: object
    negative: object
    latent_image: dict[str, Any]
    denoise: float


class ContextualArguments(TypedDict):
    """Describe global context and the sole local tile plan's execution controls."""

    diffusion_mode: str
    latent_context_size: int
    latent_context_overlap: int
    latent_context_batch_size: int
    global_weight: float
    global_steps: int
    global_decay: float


class TiledArguments(TypedDict):
    """Describe local geometry and mask-dependent denoising policy."""

    diffusion_mode: str
    latent_tile_width: int
    latent_tile_height: int
    latent_tile_overlap: int
    latent_tile_batch_size: int
    differential_diffusion: bool


class SamplerOptionsSamplingService:
    """Route configuration independently of connection order or node placement."""

    def sample(
        self,
        *,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: object,
        negative: object,
        latent_image: dict[str, Any],
        denoise: float,
        options: SamplerOptions | None = None,
    ) -> dict[str, Any]:
        """Apply regional attention once, then one spatial authority and inversion."""
        if options is not None and not isinstance(options, SamplerOptions):
            raise TypeError(
                "KSampler options must come from SimpleSyrup options nodes."
            )
        configured = options if options is not None else SamplerOptions()
        context = configured.contextual_diffusion
        tiling = configured.tiling
        if context is not None and tiling is None:
            tiling = TilingOptions(
                width=context.context_size,
                height=context.context_size,
                overlap=min(32, context.context_size - 1),
            )
        arguments: SamplingArguments = {
            "model": model,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "positive": positive,
            "negative": negative,
            "latent_image": latent_image,
            "denoise": denoise,
        }
        self._preflight(arguments, configured, tiling)
        attention = configured.attention_coupling
        inversion = configured.noise_inversion
        if context is not None:
            assert tiling is not None
            contextual_arguments: ContextualArguments = {
                "diffusion_mode": tiling.diffusion_mode,
                "latent_context_size": context.context_size,
                "latent_context_overlap": tiling.overlap,
                "latent_context_batch_size": tiling.batch_size,
                "global_weight": context.global_weight,
                "global_steps": context.global_steps,
                "global_decay": context.global_decay,
            }
            if attention is not None:
                result = ContextualAttentionCouplingSamplingService().sample(
                    **arguments,
                    **contextual_arguments,
                    region_masks=attention.region_masks,
                    regional_prompt_weight=attention.regional_prompt_weight,
                    region_mask_feather=attention.region_mask_feather,
                    segs=tiling.segs,
                    tiling=tiling,
                    noise_inversion=inversion,
                )
            else:
                result = ContextualDiffusionSamplingService().sample(
                    **arguments,
                    **contextual_arguments,
                    segs=tiling.segs,
                    tiling=tiling,
                    noise_inversion=inversion,
                )
            return result.latent
        if tiling is not None:
            tiled_arguments: TiledArguments = {
                "diffusion_mode": tiling.diffusion_mode,
                "latent_tile_width": tiling.width,
                "latent_tile_height": tiling.height,
                "latent_tile_overlap": tiling.overlap,
                "latent_tile_batch_size": tiling.batch_size,
                "differential_diffusion": tiling.differential_diffusion,
            }
            if attention is not None:
                return TiledAttentionCouplingSamplingService().sample(
                    **arguments,
                    **tiled_arguments,
                    region_masks=attention.region_masks,
                    regional_prompt_weight=attention.regional_prompt_weight,
                    region_mask_feather=attention.region_mask_feather,
                    segs=tiling.segs,
                    noise_inversion=inversion,
                )
            return TiledDiffusionSamplingService().sample(
                **arguments,
                **tiled_arguments,
                segs=tiling.segs,
                noise_inversion=inversion,
            )
        if attention is not None:
            return AttentionCouplingSamplingService().sample(
                **arguments,
                region_masks=attention.region_masks,
                regional_prompt_weight=attention.regional_prompt_weight,
                region_mask_feather=attention.region_mask_feather,
                noise_inversion=inversion,
            )
        return KSamplerSamplingService().sample(**arguments, noise_inversion=inversion)

    def _preflight(
        self,
        arguments: SamplingArguments,
        options: SamplerOptions,
        tiling: TilingOptions | None,
    ) -> None:
        """Reject unsupported schedules and endpoints before preparing models."""
        samples = arguments["latent_image"].get("samples")
        if not isinstance(samples, torch.Tensor):
            raise TypeError("KSampler latent samples must be a torch.Tensor.")
        validate_sampling_controls(
            steps=arguments["steps"],
            denoise=arguments["denoise"],
            latent_tile_width=tiling.width if tiling is not None else 16,
            latent_tile_height=tiling.height if tiling is not None else 16,
            latent_tile_batch_size=tiling.batch_size if tiling is not None else 1,
        )
        incompatible_unipc = options.contextual_diffusion is not None or (
            tiling is not None and tiling.diffusion_mode == "multidiffusion"
        )
        if incompatible_unipc and arguments["sampler_name"] in {"uni_pc", "uni_pc_bh2"}:
            raise ValueError(
                "Tiling and Contextual Diffusion do not support UniPC samplers."
            )
        if options.noise_inversion is not None:
            view = (
                sampling_schedulers.SchedulerView(tiling.width, tiling.height)
                if tiling is not None
                else sampling_schedulers.SchedulerView.from_tensor(samples)
            )
            sigmas = sampling_schedulers.calculate_sigmas(
                model=arguments["model"],
                scheduler_name=arguments["scheduler"],
                sampler_name=arguments["sampler_name"],
                steps=arguments["steps"],
                denoise=arguments["denoise"],
                view=view,
            )
            validate_inversion_target(arguments["model"], sigmas)
