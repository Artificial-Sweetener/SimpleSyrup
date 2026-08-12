# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Native Comfy v3 node for Contextual Attention Coupling."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..nodes import tooltips
from ..services.contextual_attention_coupling_sampling_service import (
    ContextualAttentionCouplingSamplingService,
)
from .ksampler_schema import (
    attention_coupling_ksampler_inputs,
    contextual_diffusion_inputs,
)

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class KSamplerContextualAttentionCouplingV3(_ComfyNodeBase):
    """Sample contextual model views with regional attention."""

    sampling_service_class: ClassVar[
        type[ContextualAttentionCouplingSamplingService]
    ] = ContextualAttentionCouplingSamplingService

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the Contextual Attention Coupling sampler schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.KSamplerAttentionCouplingContextual",
            display_name="KSampler (Attention Coupling + Contextual Diffusion)",
            category="SimpleSyrup/Sampling",
            description=(
                "Preserves large-image composition through Contextual Diffusion "
                "while coupling regional attention in every local and reduced-global "
                "Anima or standard SD/SDXL view. Global LoRAs remain on the input "
                "model. Anima regional LoRA stacks are prepared once, retain "
                "independent schedules and full quality, and skip inactive work. "
                "Optional SEGS guide the shared local tile plan. Standard SD/SDXL "
                "regional model-side hooks and unsupported Anima targets fail before "
                "sampling."
            ),
            search_aliases=[
                "contextual attention coupling",
                "contextual regional lora",
                "anima contextual regional prompt",
                "sdxl contextual regional prompt",
                "contextual multidiffusion regional lora",
                "contextual mixture of diffusers regional lora",
            ],
            inputs=[
                *attention_coupling_ksampler_inputs(_comfy_io),
                *contextual_diffusion_inputs(_comfy_io),
                _comfy_io.SEGS.Input(
                    "segs",
                    optional=True,
                    tooltip=tooltips.CONTEXTUAL_DIFFUSION_SEGS,
                ),
            ],
            outputs=[
                _comfy_io.Latent.Output(
                    "latent",
                    tooltip=tooltips.DENOISED_LATENT_OUTPUT,
                ),
                _comfy_io.SEGS.Output(
                    "contexts_segs",
                    tooltip=tooltips.CONTEXTUAL_DIFFUSION_CONTEXTS_OUTPUT,
                ),
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
        region_masks: object,
        regional_prompt_weight: float,
        region_mask_feather: int,
        latent_image: dict[str, Any],
        denoise: float = 1.0,
        diffusion_mode: str = "multidiffusion",
        latent_context_size: int = 96,
        latent_context_overlap: int = 32,
        latent_context_batch_size: int = 4,
        global_weight: float = 1.0,
        global_steps: int = 1,
        global_decay: float = 0.5,
        segs: object | None = None,
    ) -> tuple[dict[str, Any], object]:
        """Delegate the complete request to the combined application service."""

        result = cls.sampling_service_class().sample(
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
            diffusion_mode=diffusion_mode,
            latent_context_size=latent_context_size,
            latent_context_overlap=latent_context_overlap,
            latent_context_batch_size=latent_context_batch_size,
            global_weight=global_weight,
            global_steps=global_steps,
            global_decay=global_decay,
            segs=segs,
        )
        return result.latent, result.contexts
