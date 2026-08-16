# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Translate one pinned P5.7 definition into repeated PRIMARY_ADAPTER adapter uses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_module_surface import AnimaModuleSurface
from simple_syrup.runtime.regional_lora.anima_targets import AnimaLoraAdmission
from simple_syrup.runtime.regional_lora.execution_cache import (
    RegionalLoraExecutionCache,
)

from .manifest import PerformanceProfile
from .runtime_profile import (
    PerformanceRegionalAdapterUse,
    PerformanceRuntimeProfile,
    derive_runtime_profile,
)


def build_runtime_profile(
    definition: PerformanceProfile,
    *,
    source: Any,
    surface: AnimaModuleSurface,
    attention: AnimaRegionalAttentionExecution,
    admission: AnimaLoraAdmission,
    lora_path: Path,
) -> PerformanceRuntimeProfile:
    """Build one derived clone without loading or timing it."""

    cache = RegionalLoraExecutionCache()
    adapter_uses = tuple(
        PerformanceRegionalAdapterUse(
            RegionalLoraAdapterPlan(
                adapter_identity=RegionalLoraAdapterIdentity(
                    f"{lora_path}#performance-{index}"
                ),
                composition_index=index,
                region_index=0,
                branch=RegionalLoraBranch.POSITIVE,
                model_strength=1.0 / definition.adapter_count,
                schedule=(RegionalLoraScheduleBoundary(0.0, 1.0, 1.0, 0),),
            ),
            admission,
            cache,
        )
        for index in range(definition.adapter_count)
    )
    return derive_runtime_profile(
        definition,
        source=source,
        surface=surface,
        attention=attention,
        adapter_uses=adapter_uses,
        operation=f"P5.7 {definition.profile_id} performance profile",
    )
