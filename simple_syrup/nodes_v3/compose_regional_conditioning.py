# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node for composing standard masked regional conditioning."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..services.regional_conditioning_service import RegionalConditioningService
from .ksampler_schema import regional_conditioning_inputs

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class ComposeRegionalConditioningV3(_ComfyNodeBase):
    """Compose global-first batches and ordered masks for native samplers."""

    conditioning_service_class: ClassVar[type[RegionalConditioningService]] = (
        RegionalConditioningService
    )

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the standard regional-conditioning composition schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.ComposeRegionalConditioning",
            display_name="Compose Regional Conditioning",
            category="SimpleSyrup/Conditioning",
            description=(
                "Pairs global-first prompt batches with ordered masks and returns "
                "standard masked conditioning for native Comfy samplers."
            ),
            search_aliases=["regional prompt", "masked conditioning"],
            inputs=regional_conditioning_inputs(_comfy_io),
            outputs=[
                _comfy_io.Conditioning.Output(
                    "positive",
                    tooltip=(
                        "Standard masked positive conditioning with hooks preserved."
                    ),
                ),
                _comfy_io.Conditioning.Output(
                    "negative",
                    tooltip=(
                        "Standard masked negative conditioning with hooks preserved."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive: object,
        negative: object,
        region_masks: object,
        regional_prompt_weight: float,
        region_mask_feather: int,
    ) -> tuple[object, object]:
        """Return hook-preserving standard Comfy conditioning values."""

        return cls.conditioning_service_class().assemble(
            positive=positive,
            negative=negative,
            masks=region_masks,
            regional_prompt_weight=regional_prompt_weight,
            region_mask_feather=region_mask_feather,
        )
