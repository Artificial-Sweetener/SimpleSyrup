"""Verify generic Anima benchmark runtime derivation boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID, uuid4

import comfy.sampler_helpers
import pytest
from regional_lora_test_values import single_target_execution

from simple_syrup.domain.regional_lora_plan import RegionalLoraBranch
from simple_syrup.runtime.regional_lora.anima_composition import (
    AnimaRegionalLoraComposition,
)
from simple_syrup.runtime.regional_lora.anima_module_surface import AnimaModuleSurface
from simple_syrup.runtime.regional_lora.anima_targets import AnimaLoraTargetFamily
from simple_syrup.runtime.regional_lora.execution_cache import ModelCloneLineage
from tools.anima_regional_lora_performance import runtime_profile as runtime_module
from tools.anima_regional_lora_performance.manifest import PerformanceProfile
from tools.anima_regional_lora_performance.runtime_profile import (
    PerformanceRegionalAdapterUse,
)


@dataclass(frozen=True, slots=True)
class _SourceModel:
    """Expose the immutable Comfy clone identity used by execution caches."""

    clone_base_uuid: UUID
    patches_uuid: UUID


@dataclass(slots=True)
class _DerivedModel:
    """Expose only model options required by profile preparation."""

    model_options: dict[str, object]


class _PatcherLifecycle:
    """Capture profile derivation without requiring an installed Anima graph."""

    def __init__(self, derived: _DerivedModel) -> None:
        """Retain the derived value returned to the runtime builder."""

        self.derived = derived
        self.source: object | None = None
        self.operation: str | None = None

    def derive_model(
        self,
        source: object,
        mutations: object,
        *,
        operation: str,
    ) -> _DerivedModel:
        """Capture the source and return the configured derived model."""

        assert mutations == ()
        self.source = source
        self.operation = operation
        return self.derived


def test_runtime_profile_passes_lineage_and_attention_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve typed execution ownership through generic model derivation."""

    fixture = single_target_execution(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    source = _SourceModel(uuid4(), uuid4())
    derived = _DerivedModel({"transformer_options": {}})
    lifecycle = _PatcherLifecycle(derived)
    captured: list[AnimaRegionalLoraComposition | None] = []

    def override(function: Any, *args: Any, **kwargs: Any) -> Any:
        """Delegate through the benchmark override contract."""

        return function(*args, **kwargs)

    surface = AnimaModuleSurface(cast(Any, object()), (), ())

    def mutations(
        selected_surface: AnimaModuleSurface,
        attention: object,
        *,
        composition: AnimaRegionalLoraComposition | None,
    ) -> tuple[()]:
        """Capture the exact composition passed to the mutation owner."""

        assert selected_surface is surface
        assert attention is fixture.attention
        captured.append(composition)
        return ()

    def prepare_model_patcher(
        model: object,
        noise_shape: object,
        model_options: object,
    ) -> None:
        """Verify the derived model receives copied options and override."""

        assert model is derived
        assert noise_shape == {}
        assert model_options == {
            "transformer_options": {
                "optimized_attention_override": override,
            }
        }

    monkeypatch.setattr(runtime_module, "PATCHER_LIFECYCLE", lifecycle)
    monkeypatch.setattr(
        runtime_module,
        "anima_attention_coupling_mutations",
        mutations,
    )
    monkeypatch.setattr(
        comfy.sampler_helpers,
        "prepare_model_patcher",
        prepare_model_patcher,
    )

    result = runtime_module.derive_runtime_profile(
        PerformanceProfile("regional-lora-1", 1, 15.0),
        source=source,
        surface=surface,
        attention=fixture.attention,
        adapter_uses=(
            PerformanceRegionalAdapterUse(
                fixture.adapter_plan,
                fixture.admission,
                fixture.cache,
            ),
        ),
        operation="runtime profile test",
        optimized_attention_override=override,
    )

    assert lifecycle.source is source
    assert lifecycle.operation == "runtime profile test"
    assert result.model is derived
    assert result.executions[0].model_lineage == ModelCloneLineage.from_model(source)
    assert result.caches == (fixture.cache,)
    assert result.composition is captured[0]


def test_runtime_profile_rejects_declared_adapter_count_mismatch() -> None:
    """Fail before model mutation when declaration and adapter uses diverge."""

    fixture = single_target_execution(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )

    with pytest.raises(ValueError, match="must match the declared count"):
        runtime_module.derive_runtime_profile(
            PerformanceProfile("invalid", 0, 0.0),
            source=_SourceModel(uuid4(), uuid4()),
            surface=AnimaModuleSurface(cast(Any, object()), (), ()),
            attention=fixture.attention,
            adapter_uses=(
                PerformanceRegionalAdapterUse(
                    fixture.adapter_plan,
                    fixture.admission,
                    fixture.cache,
                ),
            ),
            operation="invalid runtime profile",
        )
