# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify cache admission against regional execution identity changes."""

from __future__ import annotations

from uuid import uuid4

import pytest
import torch

from simple_syrup.domain.conditioning_schedule import ConditioningScheduleRange
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_attention_execution import (
    RegionalAttentionExecutionMode,
)
from simple_syrup.domain.regional_lora_plan import (
    EMPTY_REGIONAL_LORA_PLAN,
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraPlan,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.domain.regional_model_capabilities import RegionalModelFamily
from simple_syrup.runtime.regional_model_patch_interop import (
    RegionalModelPatchInteropReport,
    RegionalModelPatchInteropValidator,
    RegionalPreservedModelModifier,
)


def test_static_full_context_cache_is_admitted() -> None:
    """Preserve one complete cache when regional execution identity is fixed."""

    plan = _processed_plan()

    RegionalModelPatchInteropValidator().validate_execution(
        _cache_report(RegionalPreservedModelModifier.EASYCACHE),
        plan,
        RegionalAttentionExecutionMode.FULL,
    )

    assert plan.is_time_invariant


@pytest.mark.parametrize(
    "mode",
    [RegionalAttentionExecutionMode.TILED, RegionalAttentionExecutionMode.CONTEXTUAL],
)
def test_cache_is_rejected_when_spatial_view_identity_changes(
    mode: RegionalAttentionExecutionMode,
) -> None:
    """Reject cache keys that cannot distinguish equal-shaped spatial views."""

    with pytest.raises(ValueError, match=f"{mode.value} spatial views"):
        RegionalModelPatchInteropValidator().validate_execution(
            _cache_report(RegionalPreservedModelModifier.EASYCACHE),
            _processed_plan(),
            mode,
        )


def test_cache_is_rejected_when_conditioning_schedule_changes() -> None:
    """Reject an inner regional context transition hidden from the cache key."""

    scheduled = ConditioningScheduleRange(0.25, 1.0, 75.0, 0.0)
    plan = _processed_plan(regional_schedule=scheduled)

    with pytest.raises(ValueError, match="scheduled regional execution"):
        RegionalModelPatchInteropValidator().validate_execution(
            _cache_report(RegionalPreservedModelModifier.EASYCACHE),
            plan,
            RegionalAttentionExecutionMode.FULL,
        )

    assert not plan.is_time_invariant


def test_cache_is_rejected_when_regional_lora_strength_changes() -> None:
    """Reject an inner regional LoRA transition hidden from the cache key."""

    plan = _processed_plan(lora_plan=_scheduled_lora_plan())

    with pytest.raises(ValueError, match="scheduled regional execution"):
        RegionalModelPatchInteropValidator().validate_execution(
            _cache_report(RegionalPreservedModelModifier.EASYCACHE),
            plan,
            RegionalAttentionExecutionMode.FULL,
        )

    assert not plan.is_time_invariant


@pytest.mark.parametrize("mode", list(RegionalAttentionExecutionMode))
def test_non_cache_modifier_is_admitted_for_every_execution_mode(
    mode: RegionalAttentionExecutionMode,
) -> None:
    """Keep execution-mode policy scoped to installed cache owners."""

    report = RegionalModelPatchInteropReport(
        RegionalModelFamily.ANIMA,
        (RegionalPreservedModelModifier.OPTIMIZED_ATTENTION_OVERRIDE,),
    )

    RegionalModelPatchInteropValidator().validate_execution(
        report,
        _processed_plan(regional_schedule=_bounded_schedule()),
        mode,
    )


def test_time_invariance_values_are_owned_by_schedule_domains() -> None:
    """Recognize full authored conditioning and constant-strength LoRA schedules."""

    assert ConditioningScheduleRange(None, None, None, None).is_time_invariant
    assert ConditioningScheduleRange(0.0, 1.0, 100.0, 0.0).is_time_invariant
    assert not ConditioningScheduleRange(None, None, 100.0, None).is_time_invariant
    assert _constant_lora_plan().is_time_invariant
    assert not _scheduled_lora_plan().is_time_invariant


def _cache_report(
    modifier: RegionalPreservedModelModifier,
) -> RegionalModelPatchInteropReport:
    """Return one structurally admitted cache report."""

    return RegionalModelPatchInteropReport(RegionalModelFamily.ANIMA, (modifier,))


def _processed_plan(
    *,
    regional_schedule: ConditioningScheduleRange | None = None,
    lora_plan: RegionalLoraPlan = EMPTY_REGIONAL_LORA_PLAN,
) -> ProcessedRegionalAttentionPlan:
    """Return one canonical one-region processed plan."""

    schedule = regional_schedule or ConditioningScheduleRange(None, None, None, None)
    masks = torch.ones((1, 2, 2))
    return ProcessedRegionalAttentionPlan(
        _branch(1.0, 2.0, schedule),
        _branch(-1.0, -2.0, schedule),
        RegionalMaskBank(masks, masks.clone(), 2, 2),
        lora_plan,
    )


def _branch(
    base_value: float,
    regional_value: float,
    regional_schedule: ConditioningScheduleRange,
) -> ProcessedRegionalAttentionBranch:
    """Return one base context and one regional context."""

    unbounded = ConditioningScheduleRange(None, None, None, None)
    return ProcessedRegionalAttentionBranch(
        _context(0, None, base_value, unbounded),
        (_context(1, 0, regional_value, regional_schedule),),
    )


def _context(
    conditioning_index: int,
    region_index: int | None,
    value: float,
    schedule: ConditioningScheduleRange,
) -> ProcessedRegionalAttentionContext:
    """Return one model-ready conditioning context."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        (
            ProcessedRegionalAttentionEntry(
                0,
                uuid4(),
                schedule,
                torch.full((1, 2, 3), value),
                1.0,
            ),
        ),
    )


def _bounded_schedule() -> ConditioningScheduleRange:
    """Return one scheduled regional conditioning range."""

    return ConditioningScheduleRange(0.25, 0.75, 75.0, 25.0)


def _constant_lora_plan() -> RegionalLoraPlan:
    """Return one adapter whose keyframes retain one effective strength."""

    return _lora_plan(
        (
            RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),
            RegionalLoraScheduleBoundary(0.5, 50.0, 1.0, 1),
        )
    )


def _scheduled_lora_plan() -> RegionalLoraPlan:
    """Return one adapter whose effective strength changes during sampling."""

    return _lora_plan(
        (
            RegionalLoraScheduleBoundary(0.0, 100.0, 0.0, 0),
            RegionalLoraScheduleBoundary(0.5, 50.0, 1.0, 1),
        )
    )


def _lora_plan(
    schedule: tuple[RegionalLoraScheduleBoundary, ...],
) -> RegionalLoraPlan:
    """Return one canonical regional adapter plan."""

    return RegionalLoraPlan(
        (
            RegionalLoraAdapterPlan(
                RegionalLoraAdapterIdentity("adapter"),
                0,
                0,
                RegionalLoraBranch.POSITIVE,
                1.0,
                schedule,
            ),
        )
    )
