# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derive one typed Anima benchmark runtime from explicit adapter uses."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

import comfy.sampler_helpers

from simple_syrup.domain.regional_lora_plan import RegionalLoraAdapterPlan
from simple_syrup.runtime.patcher_lifecycle import PATCHER_LIFECYCLE
from simple_syrup.runtime.regional_lora.anima_attention_coupling import (
    anima_attention_coupling_mutations,
)
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_composition import (
    AnimaRegionalLoraComposition,
)
from simple_syrup.runtime.regional_lora.anima_execution_scope import (
    AnimaRegionalLoraAdapterExecution,
)
from simple_syrup.runtime.regional_lora.anima_module_surface import AnimaModuleSurface
from simple_syrup.runtime.regional_lora.anima_targets import AnimaLoraAdmission
from simple_syrup.runtime.regional_lora.execution_cache import (
    ModelCloneLineage,
    RegionalLoraExecutionCache,
)


class PerformanceProfileDefinition(Protocol):
    """Expose the profile identity and declared adapter-use cardinality."""

    @property
    def profile_id(self) -> str:
        """Return the stable profile identity."""

        ...

    @property
    def adapter_count(self) -> int:
        """Return the declared adapter-use count."""

        ...


OptimizedAttentionOverride = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class PerformanceRegionalAdapterUse:
    """Bind one complete adapter plan to admitted tensors and its cache."""

    plan: RegionalLoraAdapterPlan
    admission: AnimaLoraAdmission
    cache: RegionalLoraExecutionCache


@dataclass(slots=True)
class PerformanceRuntimeProfile:
    """Retain one derived model and its authoritative execution state."""

    definition: PerformanceProfileDefinition
    model: Any
    model_options: dict[str, Any]
    executions: tuple[AnimaRegionalLoraAdapterExecution, ...]
    caches: tuple[RegionalLoraExecutionCache, ...]
    composition: AnimaRegionalLoraComposition | None

    @property
    def prepared_target_count(self) -> int:
        """Return exact device-prepared target count after warmup."""

        return sum(cache.size for cache in self.caches)


def derive_runtime_profile(
    definition: PerformanceProfileDefinition,
    *,
    source: Any,
    surface: AnimaModuleSurface,
    attention: AnimaRegionalAttentionExecution,
    adapter_uses: tuple[PerformanceRegionalAdapterUse, ...],
    operation: str,
    optimized_attention_override: OptimizedAttentionOverride | None = None,
) -> PerformanceRuntimeProfile:
    """Derive one model from explicit uses without inventing fixture policy."""

    _validate_profile_inputs(
        definition,
        surface=surface,
        attention=attention,
        adapter_uses=adapter_uses,
        operation=operation,
        optimized_attention_override=optimized_attention_override,
    )
    model_lineage = ModelCloneLineage.from_model(source)
    executions = tuple(
        AnimaRegionalLoraAdapterExecution(
            adapter_use.plan,
            adapter_use.admission,
            attention,
            model_lineage,
            adapter_use.cache,
        )
        for adapter_use in adapter_uses
    )
    composition = AnimaRegionalLoraComposition(executions) if executions else None
    mutations = anima_attention_coupling_mutations(
        surface,
        attention,
        composition=composition,
    )
    derived = PATCHER_LIFECYCLE.derive_model(source, mutations, operation=operation)
    model_options = deepcopy(derived.model_options)
    if optimized_attention_override is not None:
        transformer_options = model_options.setdefault("transformer_options", {})
        if not isinstance(transformer_options, dict):
            raise TypeError(
                "Anima performance transformer_options must be a dictionary."
            )
        transformer_options["optimized_attention_override"] = (
            optimized_attention_override
        )
    comfy.sampler_helpers.prepare_model_patcher(derived, {}, model_options)
    return PerformanceRuntimeProfile(
        definition=definition,
        model=derived,
        model_options=model_options,
        executions=executions,
        caches=_unique_caches(adapter_uses),
        composition=composition,
    )


def _validate_profile_inputs(
    definition: PerformanceProfileDefinition,
    *,
    surface: AnimaModuleSurface,
    attention: AnimaRegionalAttentionExecution,
    adapter_uses: tuple[PerformanceRegionalAdapterUse, ...],
    operation: str,
    optimized_attention_override: OptimizedAttentionOverride | None,
) -> None:
    """Fail closed on malformed generic benchmark construction inputs."""

    if not isinstance(definition.profile_id, str) or not definition.profile_id:
        raise ValueError("Anima performance profile id must be non-empty.")
    if (
        isinstance(definition.adapter_count, bool)
        or not isinstance(definition.adapter_count, int)
        or definition.adapter_count < 0
    ):
        raise ValueError("Anima performance adapter count must be non-negative.")
    if len(adapter_uses) != definition.adapter_count:
        raise ValueError(
            "Anima performance adapter uses must match the declared count."
        )
    if not isinstance(surface, AnimaModuleSurface):
        raise TypeError("Anima performance requires a module surface.")
    if not isinstance(attention, AnimaRegionalAttentionExecution):
        raise TypeError("Anima performance requires an attention execution.")
    if not isinstance(adapter_uses, tuple) or any(
        not isinstance(adapter_use, PerformanceRegionalAdapterUse)
        for adapter_use in adapter_uses
    ):
        raise TypeError("Anima performance adapter uses must be a typed tuple.")
    if tuple(adapter_use.plan.composition_index for adapter_use in adapter_uses) != (
        tuple(range(len(adapter_uses)))
    ):
        raise ValueError("Anima performance adapter uses must retain declared order.")
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("Anima performance operation must be non-empty.")
    if optimized_attention_override is not None and not callable(
        optimized_attention_override
    ):
        raise TypeError("Anima performance attention override must be callable.")


def _unique_caches(
    adapter_uses: tuple[PerformanceRegionalAdapterUse, ...],
) -> tuple[RegionalLoraExecutionCache, ...]:
    """Return first-seen cache owners without exposing their internal keys."""

    values: list[RegionalLoraExecutionCache] = []
    identities: set[int] = set()
    for adapter_use in adapter_uses:
        identity = id(adapter_use.cache)
        if identity not in identities:
            identities.add(identity)
            values.append(adapter_use.cache)
    return tuple(values)
