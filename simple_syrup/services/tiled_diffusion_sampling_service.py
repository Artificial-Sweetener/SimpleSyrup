# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application service for selectable tiled diffusion latent sampling."""

from __future__ import annotations

from typing import Any

import torch

from ..domain.conditioning_batch import ConditioningBatch, select_conditioning
from ..domain.segs import coerce_segs_group
from ..domain.segs_tiled_diffusion import build_segs_guided_tiled_diffusion_plan
from ..domain.tiled_diffusion import TiledDiffusionPlan, validate_tiled_diffusion_mode
from ..runtime import mixture_of_diffusers_sampling, multidiffusion_sampling
from ..runtime.detail_previews import DetailPreviewContext

Latent = dict[str, Any]


class TiledDiffusionSamplingService:
    """Route tiled diffusion sampling requests to the selected runtime."""

    def sample(
        self,
        *,
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
        preview_context: DetailPreviewContext | None = None,
        differential_diffusion: bool = False,
        allow_full_context_masks: bool = False,
        segs: object | None = None,
    ) -> Latent:
        """Sample a latent with the selected tiled diffusion method."""

        validate_tiled_diffusion_mode(diffusion_mode)
        if segs is not None:
            return self._sample_segs_guided(
                diffusion_mode=diffusion_mode,
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
                latent_tile_width=latent_tile_width,
                latent_tile_height=latent_tile_height,
                latent_tile_overlap=latent_tile_overlap,
                latent_tile_batch_size=latent_tile_batch_size,
                preview_context=preview_context,
                differential_diffusion=differential_diffusion,
                allow_full_context_masks=allow_full_context_masks,
                segs=segs,
            )
        if self._uses_conditioning_batch(positive, negative):
            return self._sample_conditioning_batch(
                diffusion_mode=diffusion_mode,
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
                latent_tile_width=latent_tile_width,
                latent_tile_height=latent_tile_height,
                latent_tile_overlap=latent_tile_overlap,
                latent_tile_batch_size=latent_tile_batch_size,
                preview_context=preview_context,
                differential_diffusion=differential_diffusion,
                allow_full_context_masks=allow_full_context_masks,
            )
        return self._sample_single(
            diffusion_mode=diffusion_mode,
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
            latent_tile_width=latent_tile_width,
            latent_tile_height=latent_tile_height,
            latent_tile_overlap=latent_tile_overlap,
            latent_tile_batch_size=latent_tile_batch_size,
            preview_context=preview_context,
            differential_diffusion=differential_diffusion,
            allow_full_context_masks=allow_full_context_masks,
        )

    def _sample_single(
        self,
        *,
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
        allow_full_context_masks: bool,
        tiled_plan: TiledDiffusionPlan | None = None,
    ) -> Latent:
        """Route one single-latent tiled sample to its selected runtime."""

        if diffusion_mode == "multidiffusion":
            return multidiffusion_sampling.sample_multidiffusion(
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
                latent_tile_width=latent_tile_width,
                latent_tile_height=latent_tile_height,
                latent_tile_overlap=latent_tile_overlap,
                latent_tile_batch_size=latent_tile_batch_size,
                preview_context=preview_context,
                differential_diffusion=differential_diffusion,
                allow_full_context_masks=allow_full_context_masks,
                tiled_plan=tiled_plan,
            )
        return mixture_of_diffusers_sampling.sample_mixture_of_diffusers(
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
            latent_tile_width=latent_tile_width,
            latent_tile_height=latent_tile_height,
            latent_tile_overlap=latent_tile_overlap,
            latent_tile_batch_size=latent_tile_batch_size,
            preview_context=preview_context,
            differential_diffusion=differential_diffusion,
            allow_full_context_masks=allow_full_context_masks,
            tiled_plan=tiled_plan,
        )

    def _sample_segs_guided(
        self,
        *,
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
        allow_full_context_masks: bool,
        segs: object,
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
            item_latent = self._single_item_latent(latent_image, index)
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
            output = self._sample_single(
                diffusion_mode=diffusion_mode,
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=select_conditioning(positive, index)
                if isinstance(positive, ConditioningBatch)
                else positive,
                negative=select_conditioning(negative, index)
                if isinstance(negative, ConditioningBatch)
                else negative,
                latent_image=item_latent,
                denoise=denoise,
                latent_tile_width=latent_tile_width,
                latent_tile_height=latent_tile_height,
                latent_tile_overlap=latent_tile_overlap,
                latent_tile_batch_size=latent_tile_batch_size,
                preview_context=preview_context,
                differential_diffusion=differential_diffusion,
                allow_full_context_masks=allow_full_context_masks,
                tiled_plan=plan,
            )
            output_samples = output["samples"]
            if not isinstance(output_samples, torch.Tensor):
                raise TypeError(
                    "Tiled diffusion output samples must be a torch.Tensor."
                )
            outputs.append(output_samples)
        result = latent_image.copy()
        result.pop("downscale_ratio_spacial", None)
        result["samples"] = torch.cat(outputs, dim=0)
        return result

    def _sample_conditioning_batch(
        self,
        *,
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
        allow_full_context_masks: bool,
    ) -> Latent:
        """Sample latent batch items one at a time with selected conditioning."""

        latent_samples = latent_image["samples"]
        if not isinstance(latent_samples, torch.Tensor):
            raise TypeError("Tiled diffusion latent samples must be a torch.Tensor.")

        outputs: list[torch.Tensor] = []
        for index in range(int(latent_samples.shape[0])):
            item_latent = self._single_item_latent(latent_image, index)
            output = self.sample(
                diffusion_mode=diffusion_mode,
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=select_conditioning(positive, index),
                negative=select_conditioning(negative, index),
                latent_image=item_latent,
                denoise=denoise,
                latent_tile_width=latent_tile_width,
                latent_tile_height=latent_tile_height,
                latent_tile_overlap=latent_tile_overlap,
                latent_tile_batch_size=latent_tile_batch_size,
                preview_context=preview_context,
                differential_diffusion=differential_diffusion,
                allow_full_context_masks=allow_full_context_masks,
            )
            output_samples = output["samples"]
            if not isinstance(output_samples, torch.Tensor):
                raise TypeError(
                    "Tiled diffusion output samples must be a torch.Tensor."
                )
            outputs.append(output_samples)

        result = latent_image.copy()
        result.pop("downscale_ratio_spacial", None)
        result["samples"] = torch.cat(outputs, dim=0)
        return result

    def _single_item_latent(self, latent_image: Latent, index: int) -> Latent:
        """Return a latent dictionary for one batch item."""

        latent_samples = latent_image["samples"]
        if not isinstance(latent_samples, torch.Tensor):
            raise TypeError("Tiled diffusion latent samples must be a torch.Tensor.")
        item = latent_image.copy()
        item["samples"] = latent_samples[index : index + 1]
        if "batch_index" in item:
            item["batch_index"] = [item["batch_index"][index]]
        if "noise_mask" in item:
            item["noise_mask"] = self._slice_noise_mask(
                item["noise_mask"],
                index,
                latent_samples,
            )
        return item

    def _slice_noise_mask(
        self,
        noise_mask: Any,
        index: int,
        latent_samples: torch.Tensor,
    ) -> Any:
        """Return the noise mask slice matching one latent batch item."""

        if isinstance(noise_mask, torch.Tensor) and noise_mask.shape[0] == int(
            latent_samples.shape[0],
        ):
            return noise_mask[index : index + 1]
        return noise_mask

    def _uses_conditioning_batch(self, positive: Any, negative: Any) -> bool:
        """Return whether tiled sampling needs per-item conditioning selection."""

        return isinstance(positive, ConditioningBatch) or isinstance(
            negative,
            ConditioningBatch,
        )
