# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application service for assembling full-context regional conditioning."""

from __future__ import annotations

from typing import Any, TypeAlias, overload

import torch
from comfy.hooks import HookGroup

from ..domain.conditioning_batch import ConditioningBatch
from ..domain.regional_prompting import (
    build_regional_conditioning_plan,
    validate_regional_prompt_weight,
)
from ..masking.regional_prompt_masks import (
    prepare_regional_mask_batch,
    regional_mask,
)
from ..runtime.global_first_conditioning_hooks import (
    GLOBAL_FIRST_CONDITIONING_HOOK_COMPOSER,
)
from ..runtime.regional_conditioning_companion import detach_global_companion
from ..shared.logging import get_logger

Conditioning: TypeAlias = list[list[Any]]
ConditioningInput: TypeAlias = Conditioning | ConditioningBatch
LOGGER = get_logger(__name__)


class RegionalConditioningService:
    """Combine global and ordered regional conditioning without sampling."""

    @overload
    def assemble(
        self,
        *,
        positive: object,
        negative: None,
        masks: object,
        regional_prompt_weight: float,
        region_mask_feather: int,
    ) -> tuple[Conditioning, None]: ...

    @overload
    def assemble(
        self,
        *,
        positive: object,
        negative: ConditioningInput,
        masks: object,
        regional_prompt_weight: float,
        region_mask_feather: int,
    ) -> tuple[Conditioning, Conditioning]: ...

    @overload
    def assemble(
        self,
        *,
        positive: object,
        negative: object,
        masks: object,
        regional_prompt_weight: float,
        region_mask_feather: int,
    ) -> tuple[Conditioning, Conditioning | None]: ...

    def assemble(
        self,
        *,
        positive: object,
        negative: object,
        masks: object,
        regional_prompt_weight: float,
        region_mask_feather: int,
    ) -> tuple[Conditioning, Conditioning | None]:
        """Return masked positive and any connected negative conditioning."""

        validate_regional_prompt_weight(regional_prompt_weight)
        mask_batch = prepare_regional_mask_batch(masks, region_mask_feather)
        return self.assemble_prepared(
            positive=positive,
            negative=negative,
            mask_batch=mask_batch,
            regional_prompt_weight=regional_prompt_weight,
            region_mask_feather=region_mask_feather,
        )

    @overload
    def assemble_prepared(
        self,
        *,
        positive: object,
        negative: None,
        mask_batch: torch.Tensor,
        regional_prompt_weight: float,
        region_mask_feather: int = 0,
    ) -> tuple[Conditioning, None]: ...

    @overload
    def assemble_prepared(
        self,
        *,
        positive: object,
        negative: ConditioningInput,
        mask_batch: torch.Tensor,
        regional_prompt_weight: float,
        region_mask_feather: int = 0,
    ) -> tuple[Conditioning, Conditioning]: ...

    @overload
    def assemble_prepared(
        self,
        *,
        positive: object,
        negative: object,
        mask_batch: torch.Tensor,
        regional_prompt_weight: float,
        region_mask_feather: int = 0,
    ) -> tuple[Conditioning, Conditioning | None]: ...

    def assemble_prepared(
        self,
        *,
        positive: object,
        negative: object,
        mask_batch: torch.Tensor,
        regional_prompt_weight: float,
        region_mask_feather: int = 0,
    ) -> tuple[Conditioning, Conditioning | None]:
        """Assemble conditioning from a validated, already-feathered mask batch."""

        validate_regional_prompt_weight(regional_prompt_weight)
        prepared_mask_batch = prepare_regional_mask_batch(mask_batch, feather=0)
        assembled_positive = self._assemble_input(
            positive,
            prepared_mask_batch,
            regional_prompt_weight=regional_prompt_weight,
            input_name="positive",
        )
        assembled_negative = (
            None
            if negative is None
            else self._assemble_input(
                negative,
                prepared_mask_batch,
                regional_prompt_weight=regional_prompt_weight,
                input_name="negative",
            )
        )
        LOGGER.info(
            "Regional conditioning assembled",
            extra={
                "operation": "assemble_regional_conditioning",
                "region_count": int(prepared_mask_batch.shape[0]),
                "positive_entry_count": len(assembled_positive),
                "negative_entry_count": (
                    0 if assembled_negative is None else len(assembled_negative)
                ),
                "regional_prompt_weight": regional_prompt_weight,
                "region_mask_feather": region_mask_feather,
            },
        )
        return assembled_positive, assembled_negative

    def _assemble_input(
        self,
        value: object,
        mask_batch: torch.Tensor,
        *,
        regional_prompt_weight: float,
        input_name: str,
    ) -> Conditioning:
        """Assemble one global-first conditioning input."""

        if isinstance(value, ConditioningBatch):
            entries = value.entries
        else:
            entries = (value,)
        plan = build_regional_conditioning_plan(
            region_count=int(mask_batch.shape[0]),
            conditioning_count=len(entries),
            input_name=input_name,
        )
        global_conditioning = self._validate_conditioning(
            entries[0],
            input_name=f"{input_name} global",
        )
        if not plan.pairs:
            return self._copy_conditioning(global_conditioning)

        global_hooks = GLOBAL_FIRST_CONDITIONING_HOOK_COMPOSER.global_hooks(
            global_conditioning,
            source_label=f"{input_name} global conditioning",
        )
        hook_cache: dict[tuple[HookGroup, HookGroup], HookGroup] = {}
        assembled = self._as_default(global_conditioning)
        for pair in plan.pairs:
            conditioning = self._validate_conditioning(
                entries[pair.conditioning_index],
                input_name=(f"{input_name} regional entry {pair.conditioning_index}"),
            )
            conditioning, global_companion = detach_global_companion(conditioning)
            conditioning = GLOBAL_FIRST_CONDITIONING_HOOK_COMPOSER.compose(
                conditioning,
                global_hooks,
                source_label=(f"{input_name} regional entry {pair.conditioning_index}"),
                cache=hook_cache,
            )
            if global_companion is not None:
                global_companion = GLOBAL_FIRST_CONDITIONING_HOOK_COMPOSER.compose(
                    global_companion,
                    global_hooks,
                    source_label=(
                        f"{input_name} regional entry {pair.conditioning_index} "
                        "global companion"
                    ),
                    cache=hook_cache,
                )
            mask = regional_mask(mask_batch, pair.mask_index)
            if global_companion is not None and regional_prompt_weight < 1.0:
                assembled.extend(
                    self._with_mask(
                        global_companion,
                        mask,
                        mask_strength=1.0 - regional_prompt_weight,
                    )
                )
            if regional_prompt_weight > 0.0:
                assembled.extend(
                    self._with_mask(
                        conditioning,
                        mask,
                        mask_strength=regional_prompt_weight,
                    )
                )
        if len(assembled) == len(global_conditioning):
            return self._copy_conditioning(global_conditioning)
        return assembled

    def _as_default(self, conditioning: Conditioning) -> Conditioning:
        """Mark global conditioning to fill Comfy's remaining regional weight."""

        default_conditioning: Conditioning = []
        for item in conditioning:
            metadata = dict(item[1])
            metadata["default"] = True
            default_conditioning.append([item[0], metadata])
        return default_conditioning

    def _validate_conditioning(
        self,
        value: object,
        *,
        input_name: str,
    ) -> Conditioning:
        """Return a structurally valid standard Comfy conditioning list."""

        if not isinstance(value, list):
            raise TypeError(f"{input_name} must be a standard CONDITIONING value.")
        for index, item in enumerate(value):
            if (
                not isinstance(item, list | tuple)
                or len(item) != 2
                or not isinstance(item[1], dict)
            ):
                raise ValueError(
                    f"{input_name} item {index} must contain a tensor and metadata."
                )
        return value

    def _copy_conditioning(self, conditioning: Conditioning) -> Conditioning:
        """Copy conditioning containers and metadata without cloning tensors."""

        return [[item[0], dict(item[1])] for item in conditioning]

    def _with_mask(
        self,
        conditioning: Conditioning,
        mask: torch.Tensor,
        *,
        mask_strength: float,
    ) -> Conditioning:
        """Copy conditioning entries with one full-context regional mask."""

        masked: Conditioning = []
        for item in conditioning:
            metadata = dict(item[1])
            metadata["mask"] = mask
            metadata["mask_strength"] = mask_strength
            metadata["set_area_to_bounds"] = False
            masked.append([item[0], metadata])
        return masked
