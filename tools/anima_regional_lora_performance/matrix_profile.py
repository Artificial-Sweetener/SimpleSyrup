"""Translate one scaling position into explicit runtime adapter uses."""

from __future__ import annotations

from dataclasses import dataclass
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

from .attention_backend import SagePerformanceAttentionOverride
from .matrix_manifest import (
    PerformanceAdapterLayout,
    PerformanceAttentionBackend,
    ScalingPerformanceProfile,
)
from .runtime_profile import (
    PerformanceRegionalAdapterUse,
    PerformanceRuntimeProfile,
    derive_runtime_profile,
)


@dataclass(frozen=True, slots=True)
class PreparedScalingProfile:
    """Retain one scaling definition, runtime, surface, and backend observer."""

    definition: ScalingPerformanceProfile
    runtime: PerformanceRuntimeProfile
    surface: AnimaModuleSurface
    attention_override: SagePerformanceAttentionOverride | None


def build_scaling_runtime_profile(
    definition: ScalingPerformanceProfile,
    *,
    source: Any,
    surface: AnimaModuleSurface,
    attention: AnimaRegionalAttentionExecution,
    admission: AnimaLoraAdmission,
    lora_path: Path,
) -> PreparedScalingProfile:
    """Build one profile using only its declared matrix construction policy."""

    adapter_uses = _adapter_uses(
        definition,
        admission=admission,
        lora_path=lora_path,
    )
    attention_override = (
        SagePerformanceAttentionOverride()
        if definition.attention_backend is PerformanceAttentionBackend.SAGE
        else None
    )
    runtime = derive_runtime_profile(
        definition,
        source=source,
        surface=surface,
        attention=attention,
        adapter_uses=adapter_uses,
        operation=f"P10.1 {definition.profile_id} scaling profile",
        optimized_attention_override=attention_override,
    )
    return PreparedScalingProfile(
        definition,
        runtime,
        surface,
        attention_override,
    )


def _adapter_uses(
    definition: ScalingPerformanceProfile,
    *,
    admission: AnimaLoraAdmission,
    lora_path: Path,
) -> tuple[PerformanceRegionalAdapterUse, ...]:
    """Translate identity/region/schedule layout without executing its policy."""

    if definition.adapter_layout is PerformanceAdapterLayout.NONE:
        return ()
    cache = RegionalLoraExecutionCache()
    return tuple(
        PerformanceRegionalAdapterUse(
            RegionalLoraAdapterPlan(
                adapter_identity=RegionalLoraAdapterIdentity(
                    _adapter_identity(definition, index, lora_path)
                ),
                composition_index=index,
                region_index=_region_index(definition, index),
                branch=RegionalLoraBranch.POSITIVE,
                model_strength=_model_strength(definition),
                schedule=(
                    RegionalLoraScheduleBoundary(
                        0.0,
                        1.0,
                        (
                            0.0
                            if definition.adapter_layout
                            is PerformanceAdapterLayout.INACTIVE_STACKED
                            else 1.0
                        ),
                        0,
                    ),
                ),
            ),
            admission,
            cache,
        )
        for index in range(definition.adapter_count)
    )


def _adapter_identity(
    definition: ScalingPerformanceProfile,
    index: int,
    lora_path: Path,
) -> str:
    """Return the declared distinct or repeated immutable source identity."""

    suffix = (
        "repeated"
        if definition.adapter_layout is PerformanceAdapterLayout.REPEATED_PER_REGION
        else f"distinct-{index}"
    )
    return f"{lora_path}#matrix-{suffix}"


def _region_index(definition: ScalingPerformanceProfile, index: int) -> int:
    """Map stacked uses to region zero and regional uses by declared index."""

    if definition.adapter_layout in (
        PerformanceAdapterLayout.DISTINCT_PER_REGION,
        PerformanceAdapterLayout.REPEATED_PER_REGION,
    ):
        return index
    return 0


def _model_strength(definition: ScalingPerformanceProfile) -> float:
    """Retain unit regional strength or sum-to-one stacked strength."""

    if definition.adapter_layout in (
        PerformanceAdapterLayout.DISTINCT_PER_REGION,
        PerformanceAdapterLayout.REPEATED_PER_REGION,
    ):
        return 1.0
    return 1.0 / definition.adapter_count
