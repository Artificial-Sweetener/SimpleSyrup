# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Specify phased scene composition, specialization, and consolidation."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.runtime.regional_lora.anima_composition_phase import (
    AnimaCompositionPhaseSchedule,
    AnimaCompositionStage,
)


def test_exact_schedule_coordinates_all_three_composition_stages() -> None:
    """Establish globally, specialize regionally, then consolidate globally."""

    schedule = AnimaCompositionPhaseSchedule()
    sigmas = torch.linspace(10.0, 0.0, 11)
    phases = tuple(schedule.resolve(sigmas, sigma) for sigma in sigmas[:-1])

    assert tuple(phase.stage for phase in phases) == (
        AnimaCompositionStage.COMPOSITION,
        AnimaCompositionStage.SPECIALIZATION,
        AnimaCompositionStage.SPECIALIZATION,
        AnimaCompositionStage.SPECIALIZATION,
        AnimaCompositionStage.SPECIALIZATION,
        AnimaCompositionStage.SPECIALIZATION,
        AnimaCompositionStage.SPECIALIZATION,
        AnimaCompositionStage.CONSOLIDATION,
        AnimaCompositionStage.CONSOLIDATION,
        AnimaCompositionStage.CONSOLIDATION,
    )
    assert phases[0].regional_lora_scale == 0.0
    assert phases[0].restrict_self_attention is False
    assert phases[1].restrict_self_attention is True
    assert phases[1].regional_lora_scale == 0.25
    assert phases[2].restrict_self_attention is True
    assert phases[4].regional_lora_scale == 1.0
    assert phases[7].restrict_self_attention is False
    assert 0.85 < phases[-1].regional_lora_scale < 1.0


def test_sampler_substeps_interpolate_continuous_phase_values() -> None:
    """Avoid strength jumps for solvers that evaluate between scheduled sigmas."""

    schedule = AnimaCompositionPhaseSchedule()
    sigmas = torch.tensor([10.0, 5.0, 0.0])

    early = schedule.resolve(sigmas, 9.5)
    middle = schedule.resolve(sigmas, 6.5)
    late = schedule.resolve(sigmas, 1.5)

    assert early.stage is AnimaCompositionStage.COMPOSITION
    assert early.regional_lora_scale == 0.125
    assert middle.stage is AnimaCompositionStage.SPECIALIZATION
    assert middle.restrict_self_attention is True
    assert late.stage is AnimaCompositionStage.CONSOLIDATION
    assert late.restrict_self_attention is False
    assert 0.85 < late.regional_lora_scale < 1.0


@pytest.mark.parametrize(
    "sigmas",
    (
        torch.tensor([1.0]),
        torch.tensor([1.0, 2.0, 0.0]),
        torch.tensor([1.0, float("nan"), 0.0]),
    ),
)
def test_invalid_composition_schedules_fail_closed(sigmas: torch.Tensor) -> None:
    """Reject schedules that cannot define monotonic denoising progress."""

    with pytest.raises(ValueError):
        AnimaCompositionPhaseSchedule().resolve(sigmas, 1.0)
