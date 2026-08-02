# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application service for contextual diffusion latent sampling."""

from __future__ import annotations

from typing import Any, TypeAlias

import torch

from ..domain.conditioning_batch import ConditioningBatch, select_conditioning
from ..domain.contextual_diffusion import (
    ContextualDiffusionControls,
    build_contextual_diffusion_plan,
)
from ..domain.segs import NativeSegs, coerce_segs_group
from ..domain.tiled_diffusion import validate_tiled_diffusion_mode
from ..runtime.contextual_diffusion_sampling import sample_contextual_diffusion
from .sampling_batch import (
    combine_latent_outputs,
    latent_batch_size,
    single_item_latent,
)

Latent: TypeAlias = dict[str, Any]


class ContextualDiffusionSamplingService:
    """Plan and execute composition-preserving contextual diffusion."""

    def sample(
        self,
        *,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: Latent,
        denoise: float,
        diffusion_mode: str,
        latent_context_size: int,
        latent_context_overlap: int,
        latent_context_batch_size: int,
        global_weight: float,
        global_steps: int,
        global_decay: float,
        segs: object | None = None,
    ) -> Latent:
        """Sample a latent with global and bounded detail contexts."""

        validate_tiled_diffusion_mode(diffusion_mode)
        controls = ContextualDiffusionControls(
            latent_context_size=latent_context_size,
            latent_context_overlap=latent_context_overlap,
            latent_context_batch_size=latent_context_batch_size,
            global_weight=global_weight,
            global_steps=global_steps,
            global_decay=global_decay,
        )
        controls.validate()
        batch_size = latent_batch_size(latent_image)
        segs_group = coerce_segs_group(segs) if segs is not None else ()
        if segs_group and len(segs_group) not in (1, batch_size):
            raise ValueError(
                "Contextual Diffusion requires one SEGS payload or one per latent "
                f"batch item; received {len(segs_group)} for batch size {batch_size}."
            )
        split_batch = (
            bool(segs_group)
            or isinstance(positive, ConditioningBatch)
            or isinstance(negative, ConditioningBatch)
        )
        if not split_batch:
            return self._sample_item(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                denoise=denoise,
                diffusion_mode=diffusion_mode,
                controls=controls,
                segs=None,
            )

        outputs: list[torch.Tensor] = []
        for index in range(batch_size):
            item_output = self._sample_item(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=(
                    select_conditioning(positive, index)
                    if isinstance(positive, ConditioningBatch)
                    else positive
                ),
                negative=(
                    select_conditioning(negative, index)
                    if isinstance(negative, ConditioningBatch)
                    else negative
                ),
                latent_image=single_item_latent(latent_image, index),
                denoise=denoise,
                diffusion_mode=diffusion_mode,
                controls=controls,
                segs=(
                    segs_group[0 if len(segs_group) == 1 else index]
                    if segs_group
                    else None
                ),
            )
            samples = item_output.get("samples")
            if not isinstance(samples, torch.Tensor):
                raise TypeError(
                    "Contextual Diffusion output samples must be a torch.Tensor."
                )
            outputs.append(samples)
        return combine_latent_outputs(latent_image, outputs)

    def _sample_item(
        self,
        *,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: Latent,
        denoise: float,
        diffusion_mode: str,
        controls: ContextualDiffusionControls,
        segs: NativeSegs | None,
    ) -> Latent:
        """Build one canvas plan and execute it through the runtime adapter."""

        samples = latent_image.get("samples")
        if not isinstance(samples, torch.Tensor):
            raise TypeError(
                "Contextual Diffusion latent samples must be a torch.Tensor."
            )
        plan = build_contextual_diffusion_plan(
            latent_width=int(samples.shape[-1]),
            latent_height=int(samples.shape[-2]),
            controls=controls,
            segs=segs,
        )
        return sample_contextual_diffusion(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
            diffusion_mode=diffusion_mode,
            controls=controls,
            plan=plan,
        )
