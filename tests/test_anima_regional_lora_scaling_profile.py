"""Verify scaling definitions translate into explicit runtime adapter uses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from regional_lora_test_values import single_target_execution

from simple_syrup.domain.regional_lora_plan import RegionalLoraBranch
from simple_syrup.runtime.regional_lora.anima_module_surface import AnimaModuleSurface
from simple_syrup.runtime.regional_lora.anima_targets import AnimaLoraTargetFamily
from tools.anima_regional_lora_performance import matrix_profile as profile_module
from tools.anima_regional_lora_performance.attention_backend import (
    SagePerformanceAttentionOverride,
)
from tools.anima_regional_lora_performance.matrix_manifest import (
    default_scaling_manifest,
)
from tools.anima_regional_lora_performance.runtime_profile import (
    PerformanceRegionalAdapterUse,
    PerformanceRuntimeProfile,
)


@pytest.mark.parametrize(
    ("profile_index", "regions", "strengths", "identities", "schedule"),
    [
        (3, (0, 0, 0, 0), (0.25,) * 4, 4, 1.0),
        (7, (0, 1, 2, 3), (1.0,) * 4, 4, 1.0),
        (9, (0, 1, 2, 3), (1.0,) * 4, 1, 1.0),
        (11, (0, 0, 0, 0), (0.25,) * 4, 4, 0.0),
    ],
)
def test_scaling_profile_translates_adapter_policy(
    monkeypatch: pytest.MonkeyPatch,
    profile_index: int,
    regions: tuple[int, ...],
    strengths: tuple[float, ...],
    identities: int,
    schedule: float,
) -> None:
    """Pin distinct/repeated, stacked/regional, and inactive construction."""

    fixture = single_target_execution(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    definition = default_scaling_manifest().profiles[profile_index]
    expected = cast(PerformanceRuntimeProfile, object())
    captured: dict[str, object] = {}

    def derive_runtime_profile(
        selected_definition: object,
        **values: object,
    ) -> PerformanceRuntimeProfile:
        """Capture typed values passed to the generic runtime owner."""

        assert selected_definition is definition
        captured.update(values)
        return expected

    monkeypatch.setattr(
        profile_module,
        "derive_runtime_profile",
        derive_runtime_profile,
    )
    surface = AnimaModuleSurface(cast(Any, object()), (), ())
    result = profile_module.build_scaling_runtime_profile(
        definition,
        source=object(),
        surface=surface,
        attention=fixture.attention,
        admission=fixture.admission,
        lora_path=Path("adapter-a.safetensors"),
    )

    uses = cast(tuple[PerformanceRegionalAdapterUse, ...], captured["adapter_uses"])
    assert result.runtime is expected
    assert result.surface is surface
    assert tuple(use.plan.region_index for use in uses) == regions
    assert tuple(use.plan.model_strength for use in uses) == strengths
    assert len({use.plan.adapter_identity for use in uses}) == identities
    assert all(use.plan.schedule[0].strength_multiplier == schedule for use in uses)
    assert len({id(use.cache) for use in uses}) == 1


def test_sage_scaling_profile_installs_observed_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep Sage selection explicit and separate from PyTorch gate policy."""

    fixture = single_target_execution(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    definition = default_scaling_manifest().profiles[-1]
    expected = cast(PerformanceRuntimeProfile, object())
    captured: dict[str, object] = {}

    def derive_runtime_profile(
        selected_definition: object,
        **values: object,
    ) -> PerformanceRuntimeProfile:
        """Capture the explicit optimized-attention override."""

        assert selected_definition is definition
        captured.update(values)
        return expected

    monkeypatch.setattr(
        profile_module,
        "derive_runtime_profile",
        derive_runtime_profile,
    )
    result = profile_module.build_scaling_runtime_profile(
        definition,
        source=object(),
        surface=AnimaModuleSurface(cast(Any, object()), (), ()),
        attention=fixture.attention,
        admission=fixture.admission,
        lora_path=Path("adapter-a.safetensors"),
    )

    override = captured["optimized_attention_override"]
    assert isinstance(override, SagePerformanceAttentionOverride)
    assert result.attention_override is override
