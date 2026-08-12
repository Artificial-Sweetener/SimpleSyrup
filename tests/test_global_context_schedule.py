"""Characterize pure Contextual Diffusion global-authority scheduling."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.domain.global_context_schedule import GlobalContextSchedule


def test_schedule_requires_at_least_one_denoising_interval() -> None:
    """Reject a sigma sequence without a terminal interval."""

    with pytest.raises(ValueError, match="at least one denoising step"):
        GlobalContextSchedule(sigmas=torch.tensor([1.0]), active_steps=1, decay=0.5)


@pytest.mark.parametrize("active_steps", [-3, 0])
def test_schedule_clamps_nonpositive_active_steps(active_steps: int) -> None:
    """Disable global authority for zero or negative requested steps."""

    schedule = GlobalContextSchedule(
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
        active_steps=active_steps,
        decay=0.5,
    )

    assert schedule.scale_for(None) == 0.0


def test_schedule_clamps_to_available_steps_and_uses_nearest_sigma() -> None:
    """Select the nearest stored step and apply exact exponential decay."""

    schedule = GlobalContextSchedule(
        sigmas=torch.tensor([1.0, 0.8, 0.6, 0.4, 0.0]),
        active_steps=99,
        decay=0.5,
    )

    assert schedule.scale_for(torch.tensor([0.99, 0.1])) == 1.0
    assert schedule.scale_for(torch.tensor([0.79])) == 0.5
    assert schedule.scale_for(torch.tensor([0.61])) == 0.25
    assert schedule.scale_for(torch.tensor([0.4])) == 0.125


@pytest.mark.parametrize("timestep", [None, "sigma", torch.tensor([])])
def test_active_schedule_rejects_missing_or_empty_timestep(timestep: object) -> None:
    """Fail closed when a model call cannot be mapped to a denoising step."""

    schedule = GlobalContextSchedule(
        sigmas=torch.tensor([1.0, 0.0]),
        active_steps=1,
        decay=0.5,
    )

    with pytest.raises(ValueError, match="non-empty tensor"):
        schedule.scale_for(timestep)
