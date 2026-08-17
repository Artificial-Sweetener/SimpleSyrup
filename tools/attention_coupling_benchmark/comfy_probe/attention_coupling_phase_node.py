# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose phase-profiled Attention Coupling through a benchmark-only node."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from simple_syrup.nodes_v3.ksampler_attention_coupling import (
    KSamplerAttentionCouplingV3,
)
from simple_syrup.nodes_v3.ksampler_schema import (
    attention_coupling_ksampler_inputs,
)

from .attention_coupling_phase_profile import (
    ProfiledAttentionCouplingSamplingService,
)
from .conditioning_batch_bridge import normalize_conditioning_batch

if TYPE_CHECKING:

    class _ComfyNodeBase(KSamplerAttentionCouplingV3):
        """Type-checking base for the benchmark-only sampler."""

        pass

else:
    _ComfyNodeBase = KSamplerAttentionCouplingV3

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class ProfiledKSamplerAttentionCouplingV3(_ComfyNodeBase):
    """Run exact Attention Coupling while publishing two outer phase timings."""

    sampling_service_class: ClassVar[type[ProfiledAttentionCouplingSamplingService]] = (
        ProfiledAttentionCouplingSamplingService
    )

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the production inputs under one dev-only benchmark identity."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.ProfiledKSamplerAttentionCoupling",
            display_name="Benchmark Profiled KSampler Attention Coupling",
            category="SimpleSyrup/Benchmark",
            inputs=attention_coupling_ksampler_inputs(_comfy_io),
            outputs=[_comfy_io.Latent.Output("latent")],
            is_dev_only=True,
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
        denoise: float,
    ) -> tuple[dict[str, Any]]:
        """Bridge canonical host batches, then run the inherited exact delegate."""

        return super().execute(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=normalize_conditioning_batch(positive),
            negative=normalize_conditioning_batch(negative),
            region_masks=region_masks,
            regional_prompt_weight=regional_prompt_weight,
            region_mask_feather=region_mask_feather,
            latent_image=latent_image,
            denoise=denoise,
        )
