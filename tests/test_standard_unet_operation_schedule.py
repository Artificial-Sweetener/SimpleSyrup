# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify standard-UNet sampling-run schedule lifecycle ownership."""

from __future__ import annotations

import pytest
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


def test_schedule_tracks_progression_and_restarts_on_new_sample_authority() -> None:
    """Advance one schedule, then reset it for a distinct sample tensor."""

    schedule = StandardUnetLoraSchedule(_plan())
    sample_sigmas = torch.tensor([2.0, 1.0, 0.0])

    assert schedule.resolve(_options(sample_sigmas, 2.0)) == (1.0,)
    assert schedule.resolve(_options(sample_sigmas, 1.0)) == (0.25,)
    assert schedule.resolve(_options(sample_sigmas, 1.0)) == (0.25,)
    assert schedule.resolve(_options(torch.tensor([1.0, 0.0]), 1.0)) == (0.25,)


def test_schedule_clear_releases_cursor_without_changing_plan() -> None:
    """Start a new canonical cursor after explicit model-detach cleanup."""

    schedule = StandardUnetLoraSchedule(_plan())
    sample_sigmas = torch.tensor([2.0, 1.0, 0.0])
    assert schedule.resolve(_options(sample_sigmas, 1.0)) == (0.25,)

    schedule.clear()

    assert schedule.resolve(_options(sample_sigmas, 2.0)) == (1.0,)


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"sigmas": torch.ones(1)}, TypeError, "sample_sigmas"),
        (
            {
                "sample_sigmas": torch.ones(1),
                "sigmas": torch.ones(1, dtype=torch.int64),
            },
            TypeError,
            "sigmas",
        ),
        (
            {
                "sample_sigmas": torch.tensor([float("nan")]),
                "sigmas": torch.ones(1),
            },
            ValueError,
            "finite",
        ),
    ],
)
def test_schedule_rejects_invalid_host_metadata(
    options: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    """Fail closed before retaining malformed schedule state."""

    with pytest.raises(error, match=message):
        StandardUnetLoraSchedule(_plan()).resolve(options)


def _plan() -> RegionalLoraPlan:
    """Return one generic two-boundary adapter schedule."""

    return RegionalLoraPlan(
        (
            RegionalLoraAdapterPlan(
                RegionalLoraAdapterIdentity("adapter"),
                0,
                0,
                RegionalLoraBranch.POSITIVE,
                0.8,
                (
                    RegionalLoraScheduleBoundary(0.0, 2.0, 1.0, 0),
                    RegionalLoraScheduleBoundary(0.5, 1.0, 0.25, 0),
                ),
            ),
        )
    )


def _options(sample_sigmas: torch.Tensor, sigma: float) -> dict[str, object]:
    """Return one exact standard-UNet schedule call."""

    return {"sample_sigmas": sample_sigmas, "sigmas": torch.tensor([sigma])}
