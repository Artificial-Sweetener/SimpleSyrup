# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 KSampler for full-context Attention Coupling."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..nodes import tooltips
from ..services.attention_coupling_sampling_service import (
    AttentionCouplingSamplingService,
)
from .ksampler_schema import attention_coupling_ksampler_inputs

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class KSamplerAttentionCouplingV3(_ComfyNodeBase):
    """Sample supported models with mask-bound regional attention."""

    sampling_service_class: ClassVar[type[AttentionCouplingSamplingService]] = (
        AttentionCouplingSamplingService
    )

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the full-context Attention Coupling sampler schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.KSamplerAttentionCoupling",
            display_name="KSampler (Attention Coupling)",
            category="SimpleSyrup/Sampling",
            description=(
                "With ordinary conditioning and no masks, denoises through the "
                "normal KSampler path without Attention Coupling preparation. "
                "With conditioning batches and masks, denoises supported Anima "
                "and standard SD/SDXL models through one "
                "shared trajectory while coupling global and masked regional "
                "cross-attention. The input MODEL may carry a global LoRA. Anima "
                "regions may also carry ordered, independently scheduled Prompt "
                "Control model LoRAs whose overlapping deltas compose in declared "
                "order. Runtime scales with active adapters, ranks, and targets. "
                "Standard SD/SDXL regional model-side hooks and unsupported Anima "
                "adapter targets fail before sampling."
            ),
            search_aliases=[
                "attention coupling",
                "regional lora",
                "anima regional prompt",
                "sdxl regional prompt",
            ],
            inputs=attention_coupling_ksampler_inputs(
                _comfy_io,
                region_masks_optional=True,
            ),
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
        denoise: float,
        region_masks: object | None = None,
        regional_prompt_weight: float = 0.5,
        region_mask_feather: int = 0,
    ) -> tuple[dict[str, Any]]:
        """Delegate ordinary or regional sampling to the routing service."""

        output = cls.sampling_service_class().sample(
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
        )
        return (output,)
