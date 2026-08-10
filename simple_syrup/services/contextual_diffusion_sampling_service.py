# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application service for contextual diffusion latent sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

import torch

from ..domain.conditioning_batch import ConditioningBatch, select_conditioning
from ..domain.context_segs import (
    ContextSegs,
    context_segs_from_tile_plan,
    merge_context_segs,
)
from ..domain.contextual_diffusion import (
    ContextualDiffusionControls,
    build_contextual_diffusion_plan,
)
from ..domain.segs import NativeSegs, coerce_segs_group
from ..domain.tiled_diffusion import validate_tiled_diffusion_mode
from ..runtime.contextual_diffusion_sampling import sample_contextual_diffusion
from ..runtime.latent_geometry import decoded_image_dimensions
from .regional_sampling_preparation_service import (
    RegionalSamplingPreparationService,
)
from .sampling_batch import (
    combine_latent_outputs,
    latent_batch_size,
    single_item_latent,
)

Latent: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class ContextualDiffusionSamplingResult:
    """Return the sampled latent and lazy non-global context SEGS."""

    latent: Latent
    contexts: ContextSegs


class ContextualDiffusionSamplingService:
    """Plan and execute composition-preserving contextual diffusion."""

    regional_preparation_service_class: ClassVar[
        type[RegionalSamplingPreparationService]
    ] = RegionalSamplingPreparationService

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
        region_masks: object | None = None,
        regional_prompt_weight: float = 0.5,
        region_mask_feather: int = 0,
    ) -> ContextualDiffusionSamplingResult:
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
        regional = self.regional_preparation_service_class().prepare(
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            region_masks=region_masks,
            regional_prompt_weight=regional_prompt_weight,
            region_mask_feather=region_mask_feather,
        )
        batch_size = latent_batch_size(latent_image)
        segs_group = coerce_segs_group(segs) if segs is not None else ()
        if segs_group and len(segs_group) not in (1, batch_size):
            raise ValueError(
                "Contextual Diffusion requires one SEGS payload or one per latent "
                f"batch item; received {len(segs_group)} for batch size {batch_size}."
            )
        image_height, image_width = (
            segs_group[0][0]
            if segs_group
            else decoded_image_dimensions(
                model=model,
                latent_image=latent_image,
                latent_height=int(latent_image["samples"].shape[-2]),
                latent_width=int(latent_image["samples"].shape[-1]),
            )
        )
        split_batch = (
            bool(segs_group)
            or regional.active
            or isinstance(regional.positive, ConditioningBatch)
            or isinstance(regional.negative, ConditioningBatch)
        )
        if not split_batch:
            return self._sample_item(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=regional.positive,
                negative=regional.negative,
                latent_image=latent_image,
                denoise=denoise,
                diffusion_mode=diffusion_mode,
                controls=controls,
                segs=None,
                image_height=image_height,
                image_width=image_width,
                region_masks=regional.planning_masks,
                allow_full_context_masks=regional.active,
            )

        outputs: list[torch.Tensor] = []
        contexts: list[ContextSegs] = []
        for index in range(batch_size):
            item_result = self._sample_item(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=(
                    select_conditioning(regional.positive, index)
                    if isinstance(regional.positive, ConditioningBatch)
                    else regional.positive
                ),
                negative=(
                    select_conditioning(regional.negative, index)
                    if isinstance(regional.negative, ConditioningBatch)
                    else regional.negative
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
                image_height=image_height,
                image_width=image_width,
                region_masks=regional.planning_masks,
                allow_full_context_masks=regional.active,
            )
            samples = item_result.latent.get("samples")
            if not isinstance(samples, torch.Tensor):
                raise TypeError(
                    "Contextual Diffusion output samples must be a torch.Tensor."
                )
            outputs.append(samples)
            contexts.append(item_result.contexts)
        return ContextualDiffusionSamplingResult(
            latent=combine_latent_outputs(latent_image, outputs),
            contexts=merge_context_segs(contexts),
        )

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
        image_height: int,
        image_width: int,
        region_masks: torch.Tensor | None,
        allow_full_context_masks: bool,
    ) -> ContextualDiffusionSamplingResult:
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
            region_masks=region_masks,
        )
        latent = sample_contextual_diffusion(
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
            allow_full_context_masks=allow_full_context_masks,
        )
        return ContextualDiffusionSamplingResult(
            latent=latent,
            contexts=context_segs_from_tile_plan(
                plan.tile_plan,
                image_height=image_height,
                image_width=image_width,
            ),
        )
