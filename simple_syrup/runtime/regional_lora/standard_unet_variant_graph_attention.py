# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve native regional and packed global standard-UNet graph inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .standard_unet_variant_base_attention import StandardUnetVariantBaseAttention
from .standard_unet_variant_conditioning import (
    StandardUnetVariantConditioningResolver,
)
from .standard_unet_variant_invocation import StandardUnetVariantInvocation


@dataclass(frozen=True, slots=True)
class StandardUnetVariantGraphAttention:
    """Retain one graph's exact context and transformer options."""

    context: torch.Tensor
    transformer_options: dict[str, object]

    def __post_init__(self) -> None:
        """Require graph-compatible context and mutable option surfaces."""

        if not isinstance(self.context, torch.Tensor) or self.context.ndim != 3:
            raise ValueError("Standard UNet graph context must use BxSxD layout.")
        if not isinstance(self.transformer_options, dict):
            raise TypeError("Standard UNet graph options must be a dictionary.")


class StandardUnetVariantGraphAttentionResolver:
    """Select exact regional inputs or packed uncovered-canvas inputs."""

    def __init__(
        self,
        *,
        conditioning: StandardUnetVariantConditioningResolver,
        base_attention: StandardUnetVariantBaseAttention,
    ) -> None:
        """Retain global-context and graph-local callback authorities."""

        if not isinstance(conditioning, StandardUnetVariantConditioningResolver):
            raise TypeError("Standard UNet graph attention requires conditioning.")
        if not isinstance(base_attention, StandardUnetVariantBaseAttention):
            raise TypeError("Standard UNet graph attention requires base attention.")
        self._conditioning = conditioning
        self._base_attention = base_attention

    def regional(
        self,
        region_index: int,
        invocation: StandardUnetVariantInvocation,
    ) -> StandardUnetVariantGraphAttention:
        """Return one region's exact context and native transformer options."""

        if not isinstance(invocation, StandardUnetVariantInvocation):
            raise TypeError("Standard UNet graph attention requires an invocation.")
        return StandardUnetVariantGraphAttention(
            self._conditioning.regional_context(region_index),
            invocation.transformer_options,
        )

    def global_base(
        self,
        invocation: StandardUnetVariantInvocation,
    ) -> StandardUnetVariantGraphAttention:
        """Return packed inputs for genuinely uncovered canvas."""

        if not isinstance(invocation, StandardUnetVariantInvocation):
            raise TypeError("Standard UNet graph attention requires an invocation.")
        return self._packed(invocation)

    def _packed(
        self,
        invocation: StandardUnetVariantInvocation,
    ) -> StandardUnetVariantGraphAttention:
        """Prepare one isolated complete Attention Couple callback surface."""

        return StandardUnetVariantGraphAttention(
            self._conditioning.global_context(),
            self._base_attention.prepare(invocation.transformer_options),
        )
