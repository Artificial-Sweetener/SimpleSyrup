# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare optional regional conditioning before sampler batch interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

import torch

from ..domain.conditioning_batch import ConditioningBatch
from ..masking.regional_prompt_masks import (
    prepare_regional_masks,
    resize_regional_mask_batch,
)
from ..shared.logging import get_logger
from .regional_conditioning_service import RegionalConditioningService

Latent: TypeAlias = dict[str, Any]
LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class RegionalSamplingPreparation:
    """Carry resolved conditioning and latent-space planning masks."""

    positive: object
    negative: object
    planning_masks: torch.Tensor | None

    @property
    def active(self) -> bool:
        """Return whether regional sampling owns conditioning interpretation."""

        return self.planning_masks is not None


class RegionalSamplingPreparationService:
    """Activate and assemble regional sampling from masks plus a condition batch."""

    conditioning_service_class: ClassVar[type[RegionalConditioningService]] = (
        RegionalConditioningService
    )

    def prepare(
        self,
        *,
        positive: object,
        negative: object,
        latent_image: Latent,
        region_masks: object | None,
        regional_prompt_weight: float,
        region_mask_feather: int,
    ) -> RegionalSamplingPreparation:
        """Return unchanged inputs unless masks explicitly disambiguate a batch."""

        uses_batch = isinstance(positive, ConditioningBatch) or isinstance(
            negative,
            ConditioningBatch,
        )
        if region_masks is None or not uses_batch:
            return RegionalSamplingPreparation(
                positive=positive,
                negative=negative,
                planning_masks=None,
            )
        samples = latent_image.get("samples")
        if not isinstance(samples, torch.Tensor):
            raise TypeError("Regional sampling latent samples must be a torch.Tensor.")
        if samples.ndim < 4:
            raise ValueError(
                "Regional sampling latent samples must include height and width axes."
            )
        latent_height = int(samples.shape[-2])
        latent_width = int(samples.shape[-1])
        prepared_masks = prepare_regional_masks(region_masks, region_mask_feather)
        planning_masks = resize_regional_mask_batch(
            prepared_masks.authored,
            height=latent_height,
            width=latent_width,
        )
        conditioning_masks = resize_regional_mask_batch(
            prepared_masks.conditioning,
            height=latent_height,
            width=latent_width,
        )
        assembled_positive, assembled_negative = (
            self.conditioning_service_class().assemble_prepared(
                positive=positive,
                negative=negative,
                mask_batch=conditioning_masks,
                regional_prompt_weight=regional_prompt_weight,
                region_mask_feather=region_mask_feather,
            )
        )
        LOGGER.info(
            "Regional sampler inputs prepared",
            extra={
                "operation": "prepare_regional_sampling",
                "region_count": int(planning_masks.shape[0]),
                "latent_height": latent_height,
                "latent_width": latent_width,
            },
        )
        return RegionalSamplingPreparation(
            positive=assembled_positive,
            negative=assembled_negative,
            planning_masks=planning_masks,
        )
