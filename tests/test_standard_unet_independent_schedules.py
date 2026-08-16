# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove independent CFG-paired schedules on the standard-UNet route."""

from __future__ import annotations

import torch

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraPlan,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.runtime.regional_lora.standard_unet_lora_schedule import (
    StandardUnetLoraSchedule,
)


def test_two_region_schedules_advance_independently_across_cfg_copies() -> None:
    """Resolve opposite keyframes while each CFG pair remains identical."""

    schedule = StandardUnetLoraSchedule(_plan())
    sample_sigmas = torch.tensor([2.0, 1.0, 0.0])

    assert schedule.resolve(_options(sample_sigmas, 2.0)) == (1.0, 0.5, 1.0, 0.5)
    assert schedule.resolve(_options(sample_sigmas, 1.0)) == (0.5, 1.0, 0.5, 1.0)
    assert schedule.resolve(_options(sample_sigmas, 1.0)) == (0.5, 1.0, 0.5, 1.0)


def _plan() -> RegionalLoraPlan:
    """Return two opposite schedules repeated over positive and negative CFG."""

    early = (
        RegionalLoraScheduleBoundary(0.0, 2.0, 1.0, 0),
        RegionalLoraScheduleBoundary(0.5, 1.0, 0.5, 0),
    )
    late = (
        RegionalLoraScheduleBoundary(0.0, 2.0, 0.5, 0),
        RegionalLoraScheduleBoundary(0.5, 1.0, 1.0, 0),
    )
    declarations = (
        ("left", 0, RegionalLoraBranch.POSITIVE, early),
        ("right", 1, RegionalLoraBranch.POSITIVE, late),
        ("left", 0, RegionalLoraBranch.NEGATIVE, early),
        ("right", 1, RegionalLoraBranch.NEGATIVE, late),
    )
    return RegionalLoraPlan(
        tuple(
            RegionalLoraAdapterPlan(
                RegionalLoraAdapterIdentity(identity),
                index,
                region,
                branch,
                1.0,
                boundaries,
            )
            for index, (identity, region, branch, boundaries) in enumerate(declarations)
        )
    )


def _options(sample_sigmas: torch.Tensor, sigma: float) -> dict[str, object]:
    """Return one exact standard-UNet schedule invocation."""

    return {"sample_sigmas": sample_sigmas, "sigmas": torch.tensor([sigma])}
