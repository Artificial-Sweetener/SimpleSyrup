# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt collected regional conditioning HookGroups into one LoRA plan."""

from __future__ import annotations

from ..domain.raw_regional_attention import RawRegionalAttentionPlan
from .regional_lora_conditioning_sources import (
    REGIONAL_LORA_CONDITIONING_SOURCE_COLLECTOR,
    RegionalLoraConditioningSourceCollector,
)
from .regional_lora_plan_adapter import (
    RegionalLoraPlanAdaptation,
    RegionalLoraPlanAdapter,
)
from .regional_model_hook_selection import (
    REGIONAL_MODEL_HOOK_SELECTOR,
    RegionalModelHookSelector,
)


class RegionalLoraConditioningAdapter:
    """Delegate HookGroup collection and immutable plan adaptation."""

    def __init__(
        self,
        plan_adapter: RegionalLoraPlanAdapter | None = None,
        hook_selector: RegionalModelHookSelector | None = None,
        source_collector: RegionalLoraConditioningSourceCollector | None = None,
    ) -> None:
        """Retain the focused source and hook-to-domain owners."""

        selector = hook_selector or REGIONAL_MODEL_HOOK_SELECTOR
        self._source_collector = source_collector or (
            REGIONAL_LORA_CONDITIONING_SOURCE_COLLECTOR
            if hook_selector is None
            else RegionalLoraConditioningSourceCollector(selector)
        )
        self._plan_adapter = plan_adapter or RegionalLoraPlanAdapter(selector)

    def adapt(
        self,
        plan: RawRegionalAttentionPlan,
        *,
        model: object,
    ) -> RegionalLoraPlanAdaptation:
        """Return ordered positive then negative regional adapter uses."""

        sources = self._source_collector.collect(plan)
        return self._plan_adapter.adapt(
            tuple(source.as_plan_source() for source in sources),
            model=model,
        )


REGIONAL_LORA_CONDITIONING_ADAPTER = RegionalLoraConditioningAdapter()
