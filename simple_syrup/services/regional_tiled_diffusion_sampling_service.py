# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute region-constrained tiled diffusion across a latent batch."""

from __future__ import annotations

from typing import Any, TypeAlias

import torch

from ..domain.regional_features import RegionalCapabilityAdmission
from ..domain.regional_tiled_diffusion import (
    build_region_constrained_tiled_diffusion_plan,
)
from ..domain.segs import coerce_segs_group
from ..runtime.detail_previews import DetailPreviewContext
from .sampling_batch import combine_latent_outputs, single_item_latent
from .tiled_diffusion_item_sampling_service import TiledDiffusionItemSampler

Latent: TypeAlias = dict[str, Any]


class RegionalTiledDiffusionSamplingService:
    """Own regional plan construction and latent-batch execution."""

    def sample(
        self,
        *,
        item_sampler: TiledDiffusionItemSampler,
        region_masks: torch.Tensor,
        segs: object | None,
        diffusion_mode: str,
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
        latent_tile_width: int,
        latent_tile_height: int,
        latent_tile_overlap: int,
        latent_tile_batch_size: int,
        preview_context: DetailPreviewContext | None,
        differential_diffusion: bool,
        capability_admission: RegionalCapabilityAdmission,
    ) -> Latent:
        """Sample every latent item with one shared regional composition."""

        latent_samples = latent_image.get("samples")
        if not isinstance(latent_samples, torch.Tensor):
            raise TypeError("Tiled diffusion latent samples must be a torch.Tensor.")
        batch_size = int(latent_samples.shape[0])
        segs_group = coerce_segs_group(segs) if segs is not None else ()
        if segs_group and len(segs_group) not in (1, batch_size):
            raise ValueError(
                "Region-constrained tiled diffusion requires one SEGS payload or "
                f"one per latent batch item; received {len(segs_group)} SEGS "
                f"payloads for batch size {batch_size}."
            )

        outputs: list[torch.Tensor] = []
        for index in range(batch_size):
            item_latent = single_item_latent(latent_image, index)
            item_samples = item_latent.get("samples")
            if not isinstance(item_samples, torch.Tensor):
                raise TypeError(
                    "Tiled diffusion latent samples must be a torch.Tensor."
                )
            plan = build_region_constrained_tiled_diffusion_plan(
                region_masks=region_masks,
                segs=(
                    segs_group[0 if len(segs_group) == 1 else index]
                    if segs_group
                    else None
                ),
                latent_width=int(item_samples.shape[-1]),
                latent_height=int(item_samples.shape[-2]),
                tile_width=latent_tile_width,
                tile_height=latent_tile_height,
                overlap=latent_tile_overlap,
                tile_batch_size=latent_tile_batch_size,
            )
            output = item_sampler(
                diffusion_mode=diffusion_mode,
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=item_latent,
                denoise=denoise,
                latent_tile_width=latent_tile_width,
                latent_tile_height=latent_tile_height,
                latent_tile_overlap=latent_tile_overlap,
                latent_tile_batch_size=latent_tile_batch_size,
                preview_context=preview_context,
                differential_diffusion=differential_diffusion,
                capability_admission=capability_admission,
                tiled_plan=plan,
            )
            output_samples = output.get("samples")
            if not isinstance(output_samples, torch.Tensor):
                raise TypeError(
                    "Tiled diffusion output samples must be a torch.Tensor."
                )
            outputs.append(output_samples)
        return combine_latent_outputs(latent_image, outputs)
