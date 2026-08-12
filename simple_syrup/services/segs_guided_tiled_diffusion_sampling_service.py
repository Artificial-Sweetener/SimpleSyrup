"""Execute SEGS-guided tiled diffusion across a latent batch."""

from __future__ import annotations

from typing import Any, TypeAlias

import torch

from ..domain.conditioning_batch import ConditioningBatch, select_conditioning
from ..domain.regional_features import RegionalCapabilityAdmission
from ..domain.segs import coerce_segs_group
from ..domain.segs_tiled_diffusion import build_segs_guided_tiled_diffusion_plan
from ..runtime.detail_previews import DetailPreviewContext
from .sampling_batch import combine_latent_outputs, single_item_latent
from .tiled_diffusion_item_sampling_service import TiledDiffusionItemSampler

Latent: TypeAlias = dict[str, Any]


class SEGSGuidedTiledDiffusionSamplingService:
    """Own per-latent SEGS alignment, plan construction, and execution."""

    def sample(
        self,
        *,
        item_sampler: TiledDiffusionItemSampler,
        segs: object,
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
        """Sample every latent batch item using its connected SEGS guide."""

        segs_group = coerce_segs_group(segs)
        latent_samples = latent_image.get("samples")
        if not isinstance(latent_samples, torch.Tensor):
            raise TypeError("Tiled diffusion latent samples must be a torch.Tensor.")
        batch_size = int(latent_samples.shape[0])
        if len(segs_group) not in (1, batch_size):
            raise ValueError(
                "SEGS-guided tiled diffusion requires one SEGS payload or one per "
                f"latent batch item; received {len(segs_group)} SEGS payloads for "
                f"batch size {batch_size}."
            )

        outputs: list[torch.Tensor] = []
        for index in range(batch_size):
            item_latent = single_item_latent(latent_image, index)
            samples = item_latent["samples"]
            if not isinstance(samples, torch.Tensor):
                raise TypeError(
                    "Tiled diffusion latent samples must be a torch.Tensor."
                )
            segs_for_item = segs_group[0 if len(segs_group) == 1 else index]
            plan = build_segs_guided_tiled_diffusion_plan(
                segs=segs_for_item,
                latent_width=int(samples.shape[-1]),
                latent_height=int(samples.shape[-2]),
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
            output_samples = output["samples"]
            if not isinstance(output_samples, torch.Tensor):
                raise TypeError(
                    "Tiled diffusion output samples must be a torch.Tensor."
                )
            outputs.append(output_samples)
        return combine_latent_outputs(latent_image, outputs)
