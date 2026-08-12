"""Characterize installed Comfy conditioning schedule admission exactly."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
import torch
from comfy import samplers

from simple_syrup.domain.conditioning_schedule import ConditioningScheduleRange
from simple_syrup.domain.conditioning_schedule_selection import (
    CONDITIONING_SCHEDULE_SELECTION_POLICY,
)


@pytest.mark.parametrize("branch_name", ["positive", "negative"])
@pytest.mark.parametrize(
    ("sigma", "expected_names"),
    [
        (100.0, ("first",)),
        (75.0, ("first", "overlap")),
        (50.0, ("first", "second", "overlap", "zero_width")),
        (25.0, ("second", "overlap")),
        (0.0, ("second",)),
    ],
)
def test_installed_comfy_uses_inclusive_schedule_boundaries(
    branch_name: str,
    sigma: float,
    expected_names: tuple[str, ...],
) -> None:
    """Pin full, adjacent, overlapping, and zero-width interval admission."""

    del branch_name
    entries = (
        ("first", _entry(timestep_start=100.0, timestep_end=50.0)),
        ("second", _entry(timestep_start=50.0, timestep_end=0.0)),
        ("overlap", _entry(timestep_start=75.0, timestep_end=25.0)),
        ("zero_width", _entry(timestep_start=50.0, timestep_end=50.0)),
    )

    assert _active_names(entries, sigma=sigma) == expected_names


def test_installed_comfy_preserves_strength_and_uuid_on_active_entry() -> None:
    """Pin exact entry identity and scalar multiplier at schedule equality."""

    entry = _entry(
        timestep_start=50.0,
        timestep_end=50.0,
        strength=0.375,
    )

    admitted = samplers.get_area_and_mult(
        entry,
        torch.ones((2, 4, 3, 3)),
        torch.tensor([50.0, 50.0]),
    )

    assert admitted is not None
    assert admitted.uuid is entry["uuid"]
    assert torch.equal(admitted.mult, torch.full((2, 4, 3, 3), 0.375))


def test_installed_comfy_returns_zero_for_branches_with_no_active_entry() -> None:
    """Pin ordinary positive and negative no-active-entry accumulation."""

    model = _NoExecutionModel()
    latent = torch.ones((1, 4, 2, 2))
    inactive = _entry(timestep_start=100.0, timestep_end=75.0)

    outputs = samplers._calc_cond_batch(  # noqa: SLF001
        model,
        [[inactive], [inactive.copy()]],
        latent,
        torch.tensor([50.0]),
        {},
    )

    assert len(outputs) == 2
    assert all(torch.equal(output, torch.zeros_like(latent)) for output in outputs)
    assert model.current_patcher.prepared_sigmas == [50.0]


def test_installed_comfy_can_admit_no_entry_inside_a_schedule_gap() -> None:
    """Pin inactive gaps instead of inventing nearest-entry fallback."""

    entries = (
        ("early", _entry(timestep_start=100.0, timestep_end=75.0)),
        ("late", _entry(timestep_start=25.0, timestep_end=0.0)),
    )

    assert _active_names(entries, sigma=50.0) == ()


@pytest.mark.parametrize(
    ("start", "end", "sigma"),
    [
        (None, None, 100.0),
        (100.0, 50.0, 100.0),
        (100.0, 50.0, 50.0),
        (100.0, 50.0, 100.01),
        (100.0, 50.0, 49.99),
        (None, 25.0, 25.0),
        (None, 25.0, 24.99),
        (75.0, None, 75.0),
        (75.0, None, 75.01),
        (50.0, 50.0, 50.0),
        (50.0, 50.0, 49.99),
    ],
)
def test_domain_schedule_policy_matches_installed_comfy_admission(
    start: float | None,
    end: float | None,
    sigma: float,
) -> None:
    """Prove the focused domain predicate against the pinned host function."""

    entry = _optional_entry(timestep_start=start, timestep_end=end)
    admitted = samplers.get_area_and_mult(
        entry,
        torch.ones((1, 4, 2, 2)),
        torch.tensor([sigma]),
    )
    schedule = ConditioningScheduleRange(None, None, start, end)

    assert CONDITIONING_SCHEDULE_SELECTION_POLICY.is_active(
        schedule,
        sigma=sigma,
    ) is (admitted is not None)


class _NoExecutionPatcher:
    """Record state preparation when every condition is inactive."""

    def __init__(self) -> None:
        """Create an empty sigma record."""

        self.prepared_sigmas: list[float] = []

    def prepare_state(
        self,
        timestep: torch.Tensor,
        model_options: dict[str, object],
    ) -> None:
        """Record installed Comfy's state-preparation call."""

        del model_options
        self.prepared_sigmas.append(float(timestep[0].item()))


class _NoExecutionModel:
    """Expose only the patcher surface reached when no entry is active."""

    def __init__(self) -> None:
        """Create the minimal installed-Comfy model boundary."""

        self.current_patcher = _NoExecutionPatcher()


def _entry(
    *,
    timestep_start: float,
    timestep_end: float,
    strength: float = 1.0,
) -> dict[str, object]:
    """Build one converted Comfy entry without unrelated metadata."""

    return {
        "model_conds": {},
        "uuid": uuid4(),
        "timestep_start": timestep_start,
        "timestep_end": timestep_end,
        "strength": strength,
    }


def _optional_entry(
    *,
    timestep_start: float | None,
    timestep_end: float | None,
) -> dict[str, object]:
    """Build one converted Comfy entry with independently optional bounds."""

    entry: dict[str, object] = {"model_conds": {}, "uuid": uuid4()}
    if timestep_start is not None:
        entry["timestep_start"] = timestep_start
    if timestep_end is not None:
        entry["timestep_end"] = timestep_end
    return entry


def _active_names(
    entries: tuple[tuple[str, dict[str, object]], ...],
    *,
    sigma: float,
) -> tuple[str, ...]:
    """Return names admitted by the installed Comfy boundary in source order."""

    latent = torch.ones((1, 4, 2, 2))
    timestep = torch.tensor([sigma])
    active: list[str] = []
    for name, entry in entries:
        admitted = samplers.get_area_and_mult(entry, latent, timestep)
        if admitted is not None:
            identity = admitted.uuid
            assert isinstance(identity, UUID)
            assert identity is entry["uuid"]
            active.append(name)
    return tuple(active)
