# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 KSampler for tiled Attention Coupling."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..nodes import tooltips
from ..services.tiled_attention_coupling_sampling_service import (
    TiledAttentionCouplingSamplingService,
)
from .ksampler_schema import (
    attention_coupling_ksampler_inputs,
    tiled_diffusion_inputs,
)

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class KSamplerTiledAttentionCouplingV3(_ComfyNodeBase):
    """Sample supported models with tiled regional attention."""

    sampling_service_class: ClassVar[type[TiledAttentionCouplingSamplingService]] = (
        TiledAttentionCouplingSamplingService
    )

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the tiled Attention Coupling sampler schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.KSamplerAttentionCouplingTiled",
            display_name="KSampler (Attention Coupling + Tiled Diffusion)",
            category="SimpleSyrup/Sampling",
            description=(
                "With ordinary conditioning and no masks, uses normal tiled "
                "diffusion without Attention Coupling preparation. With conditioning "
                "batches and masks, denoises large Anima and standard SD/SDXL "
                "latents in tiles through "
                "one shared model trajectory per tile batch while coupling global "
                "and masked regional cross-attention. The input MODEL may carry "
                "global LoRAs. Anima regions may carry independently scheduled "
                "regional LoRA stacks; inactive attention and LoRA work is pruned "
                "without changing quality. MultiDiffusion or Mixture of Diffusers "
                "fuses restored tile predictions. Standard SD/SDXL regional "
                "model-side hooks and unsupported Anima targets fail before sampling."
            ),
            search_aliases=[
                "attention coupling tiled",
                "regional lora tiled",
                "anima tiled regional prompt",
                "sdxl tiled regional prompt",
                "multidiffusion regional lora",
                "mixture of diffusers regional lora",
            ],
            inputs=[
                *attention_coupling_ksampler_inputs(
                    _comfy_io,
                    region_masks_optional=True,
                ),
                *tiled_diffusion_inputs(_comfy_io),
            ],
            outputs=[
                _comfy_io.Latent.Output(
                    "latent",
                    tooltip=tooltips.DENOISED_LATENT_OUTPUT,
                )
            ],
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: object,
        negative: object,
        latent_image: dict[str, Any],
        denoise: float = 1.0,
        diffusion_mode: str = "multidiffusion",
        latent_tile_width: int = 128,
        latent_tile_height: int = 128,
        latent_tile_overlap: int = 16,
        latent_tile_batch_size: int = 4,
        region_masks: object | None = None,
        regional_prompt_weight: float = 0.5,
        region_mask_feather: int = 0,
    ) -> tuple[dict[str, Any]]:
        """Delegate ordinary or regional tiled sampling to the routing service."""

        output = cls.sampling_service_class().sample(
            diffusion_mode=diffusion_mode,
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            region_masks=region_masks,
            regional_prompt_weight=regional_prompt_weight,
            region_mask_feather=region_mask_feather,
            latent_image=latent_image,
            denoise=denoise,
            latent_tile_width=latent_tile_width,
            latent_tile_height=latent_tile_height,
            latent_tile_overlap=latent_tile_overlap,
            latent_tile_batch_size=latent_tile_batch_size,
            preview_context=None,
            differential_diffusion=False,
        )
        return (output,)
