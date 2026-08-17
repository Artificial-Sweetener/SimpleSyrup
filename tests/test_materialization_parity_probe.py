# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify materialization-parity schedule and result aggregation."""

from __future__ import annotations

import pytest

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraScheduleBoundary,
)
from tools.attention_coupling_benchmark.comfy_probe import (
    materialization_parity_probe,
    materialized_variant_comparison,
)

MaterializationParityObservation = (
    materialization_parity_probe.MaterializationParityObservation
)
MaterializationParityProbe = materialization_parity_probe.MaterializationParityProbe
MaterializationParityVariantObservation = (
    materialization_parity_probe.MaterializationParityVariantObservation
)
MaterializedVariantComparison = (
    materialized_variant_comparison.MaterializedVariantComparison
)


def test_time_invariant_schedule_values_follow_composition_order() -> None:
    """Preserve every canonical adapter's exact effective multiplier."""

    adapters = (
        _adapter(0, 0.75, starts=(0.0, 0.5)),
        _adapter(1, 1.25, starts=(0.0,)),
    )

    observed = MaterializationParityProbe._time_invariant_multipliers(adapters)

    assert observed == (0.75, 1.25)


def test_changing_schedule_fails_closed() -> None:
    """Reject a comparison that would conceal multiple materialized states."""

    adapter = RegionalLoraAdapterPlan(
        adapter_identity=RegionalLoraAdapterIdentity("fixture"),
        composition_index=0,
        region_index=0,
        branch=RegionalLoraBranch.POSITIVE,
        model_strength=1.0,
        schedule=(
            RegionalLoraScheduleBoundary(0.0, 10.0, 1.0, 0),
            RegionalLoraScheduleBoundary(0.5, 5.0, 0.5, 0),
        ),
    )

    with pytest.raises(ValueError, match="time-invariant"):
        MaterializationParityProbe._time_invariant_multipliers((adapter,))


def test_observation_aggregates_weighted_errors_and_exactness() -> None:
    """Publish totals without averaging per-variant means equally."""

    first = _comparison(elements=2, differing=1, maximum=0.5, mean=0.25, rms=0.5)
    second = _comparison(
        elements=6,
        differing=2,
        maximum=0.25,
        mean=0.125,
        rms=0.25,
        region_index=1,
    )
    observation = MaterializationParityObservation(
        selected_device="cuda:0",
        source_load_ms=1.0,
        admission_ms=2.0,
        peak_vram_bytes=3,
        variants=(
            MaterializationParityVariantObservation(0, 4.0, 5.0, 6.0, first),
            MaterializationParityVariantObservation(1, 7.0, 8.0, 9.0, second),
        ),
    )

    payload = observation.as_json_object()

    assert payload["element_count"] == 8
    assert payload["differing_element_count"] == 3
    assert payload["max_absolute_error"] == 0.5
    assert payload["mean_absolute_error"] == pytest.approx(0.15625)
    assert payload["root_mean_squared_error"] == pytest.approx((0.109375) ** 0.5)
    assert payload["exact"] is False
    variants = payload["variants"]
    assert isinstance(variants, list)
    assert variants[0]["comparison"]["exact"] is False


def _adapter(
    composition_index: int,
    multiplier: float,
    *,
    starts: tuple[float, ...],
) -> RegionalLoraAdapterPlan:
    """Build one time-invariant domain adapter fixture."""

    return RegionalLoraAdapterPlan(
        adapter_identity=RegionalLoraAdapterIdentity(f"fixture-{composition_index}"),
        composition_index=composition_index,
        region_index=composition_index,
        branch=RegionalLoraBranch.POSITIVE,
        model_strength=1.0,
        schedule=tuple(
            RegionalLoraScheduleBoundary(
                start_percent=start,
                start_sigma=10.0 - start,
                strength_multiplier=multiplier,
                guarantee_steps=0,
            )
            for start in starts
        ),
    )


def _comparison(
    *,
    elements: int,
    differing: int,
    maximum: float,
    mean: float,
    rms: float,
    region_index: int = 0,
) -> MaterializedVariantComparison:
    """Build one aggregate comparison fixture."""

    return MaterializedVariantComparison(
        region_index=region_index,
        parameter_count=1,
        element_count=elements,
        differing_element_count=differing,
        max_absolute_error=maximum,
        mean_absolute_error=mean,
        root_mean_squared_error=rms,
        max_error_parameter_path="weight",
        reference_sha256="reference",
        candidate_sha256="candidate",
    )
