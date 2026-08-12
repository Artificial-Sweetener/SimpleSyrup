"""Execute tiled diffusion once per latent conditioning batch item."""

from __future__ import annotations

from typing import Any, TypeAlias

import torch

from ..domain.conditioning_batch import select_conditioning
from ..domain.regional_features import RegionalCapabilityAdmission
from ..runtime.detail_previews import DetailPreviewContext
from .sampling_batch import combine_latent_outputs, single_item_latent
from .tiled_diffusion_item_sampling_service import TiledDiffusionItemSampler

Latent: TypeAlias = dict[str, Any]


class TiledDiffusionConditioningBatchService:
    """Own per-item conditioning selection and latent batch recombination."""

    def sample(
        self,
        *,
        item_sampler: TiledDiffusionItemSampler,
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
        """Sample each latent item with its selected conditioning values."""

        latent_samples = latent_image.get("samples")
        if not isinstance(latent_samples, torch.Tensor):
            raise TypeError("Tiled diffusion latent samples must be a torch.Tensor.")

        outputs: list[torch.Tensor] = []
        for index in range(int(latent_samples.shape[0])):
            output = item_sampler(
                diffusion_mode=diffusion_mode,
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=select_conditioning(positive, index),
                negative=select_conditioning(negative, index),
                latent_image=single_item_latent(latent_image, index),
                denoise=denoise,
                latent_tile_width=latent_tile_width,
                latent_tile_height=latent_tile_height,
                latent_tile_overlap=latent_tile_overlap,
                latent_tile_batch_size=latent_tile_batch_size,
                preview_context=preview_context,
                differential_diffusion=differential_diffusion,
                capability_admission=capability_admission,
            )
            output_samples = output.get("samples")
            if not isinstance(output_samples, torch.Tensor):
                raise TypeError(
                    "Tiled diffusion output samples must be a torch.Tensor."
                )
            outputs.append(output_samples)

        return combine_latent_outputs(latent_image, outputs)
