"""Verify pinned P5.7 definition-to-adapter-use translation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from regional_lora_test_values import single_target_execution

from simple_syrup.domain.regional_lora_plan import RegionalLoraBranch
from simple_syrup.runtime.regional_lora.anima_module_surface import AnimaModuleSurface
from simple_syrup.runtime.regional_lora.anima_targets import AnimaLoraTargetFamily
from tools.anima_regional_lora_performance import profile as profile_module
from tools.anima_regional_lora_performance.manifest import PerformanceProfile
from tools.anima_regional_lora_performance.runtime_profile import (
    PerformanceRegionalAdapterUse,
    PerformanceRuntimeProfile,
)


def test_p57_profile_builds_ordered_shared_cache_adapter_uses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep repeated-ADAPTER_A identity, strength, schedule, and cache policy local."""

    fixture = single_target_execution(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    definition = PerformanceProfile("regional-lora-4", 4, 35.0)
    source = object()
    surface = AnimaModuleSurface(cast(Any, object()), (), ())
    expected = cast(PerformanceRuntimeProfile, object())
    captured: dict[str, object] = {}

    def derive_runtime_profile(
        selected_definition: PerformanceProfile,
        **values: object,
    ) -> PerformanceRuntimeProfile:
        """Capture the P5.7 policy passed to the generic runtime owner."""

        assert selected_definition is definition
        captured.update(values)
        return expected

    monkeypatch.setattr(
        profile_module,
        "derive_runtime_profile",
        derive_runtime_profile,
    )

    result = profile_module.build_runtime_profile(
        definition,
        source=source,
        surface=surface,
        attention=fixture.attention,
        admission=fixture.admission,
        lora_path=Path("adapter-a.safetensors"),
    )

    assert result is expected
    assert captured["source"] is source
    assert captured["surface"] is surface
    assert captured["attention"] is fixture.attention
    assert captured["operation"] == "P5.7 regional-lora-4 performance profile"
    uses = cast(tuple[PerformanceRegionalAdapterUse, ...], captured["adapter_uses"])
    assert tuple(use.plan.composition_index for use in uses) == (0, 1, 2, 3)
    assert tuple(use.plan.adapter_identity.value for use in uses) == (
        "adapter-a.safetensors#performance-0",
        "adapter-a.safetensors#performance-1",
        "adapter-a.safetensors#performance-2",
        "adapter-a.safetensors#performance-3",
    )
    assert all(use.plan.region_index == 0 for use in uses)
    assert all(use.plan.branch is RegionalLoraBranch.POSITIVE for use in uses)
    assert all(use.plan.model_strength == 0.25 for use in uses)
    assert all(use.plan.schedule[0].strength_multiplier == 1.0 for use in uses)
    assert all(use.admission is fixture.admission for use in uses)
    assert len({id(use.cache) for use in uses}) == 1


def test_p57_attention_only_profile_declares_no_adapter_uses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the zero-adapter control free from cache and composition policy."""

    fixture = single_target_execution(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    captured: list[tuple[PerformanceRegionalAdapterUse, ...]] = []
    expected = cast(PerformanceRuntimeProfile, object())

    def derive_runtime_profile(
        definition: PerformanceProfile,
        **values: object,
    ) -> PerformanceRuntimeProfile:
        """Capture the exact empty adapter-use tuple."""

        assert definition.adapter_count == 0
        captured.append(
            cast(tuple[PerformanceRegionalAdapterUse, ...], values["adapter_uses"])
        )
        return expected

    monkeypatch.setattr(
        profile_module,
        "derive_runtime_profile",
        derive_runtime_profile,
    )

    result = profile_module.build_runtime_profile(
        PerformanceProfile("attention-only", 0, 0.0),
        source=object(),
        surface=AnimaModuleSurface(cast(Any, object()), (), ()),
        attention=fixture.attention,
        admission=fixture.admission,
        lora_path=Path("adapter-a.safetensors"),
    )

    assert result is expected
    assert captured == [()]
