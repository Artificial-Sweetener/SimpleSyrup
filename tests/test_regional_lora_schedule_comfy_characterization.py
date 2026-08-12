"""Characterize installed Comfy WeightHook schedule behavior exactly."""

from __future__ import annotations

import pytest
import torch
from comfy.hooks import HookKeyframe, HookKeyframeGroup, WeightHook


def test_weight_hook_switches_inclusively_and_preserves_base_strength() -> None:
    """Pin ordered keyframe selection, equality, and effective model strength."""

    hook = WeightHook(strength_model=1.5)
    hook.hook_keyframe = _group(
        (0.0, 0.0, 0),
        (0.25, 1.0, 0),
        (0.5, 0.5, 0),
    )
    _initialize(hook.hook_keyframe)
    options = {"sample_sigmas": torch.tensor([100.0, 75.0, 50.0, 0.0])}

    observed = []
    for sigma in (100.0, 75.0, 50.0, 25.0):
        hook.hook_keyframe.prepare_current_keyframe(sigma, options)
        observed.append((hook.strength, hook.strength_model))

    assert observed == [(0.0, 0.0), (1.0, 1.5), (0.5, 0.75), (0.5, 0.75)]


def test_weight_hook_ignores_repeated_current_sigma() -> None:
    """Pin one schedule step per distinct current sigma value."""

    group = _group((0.0, 1.0, 2), (0.25, 2.0, 0))
    _initialize(group)
    options = {"sample_sigmas": torch.tensor([100.0, 75.0, 50.0, 0.0])}

    first_changed = group.prepare_current_keyframe(100.0, options)
    repeated_changed = group.prepare_current_keyframe(100.0, options)
    boundary_changed = group.prepare_current_keyframe(75.0, options)
    later_changed = group.prepare_current_keyframe(50.0, options)

    assert (first_changed, repeated_changed, boundary_changed, later_changed) == (
        False,
        False,
        False,
        True,
    )
    assert group.strength == 2.0


def test_weight_hook_drops_guarantee_before_a_truncated_sampling_range() -> None:
    """Pin effective guarantee handling against the sample schedule maximum."""

    group = _group((0.0, 1.0, 5), (0.25, 2.0, 0), (0.5, 3.0, 0))
    _initialize(group)

    changed = group.prepare_current_keyframe(
        50.0,
        {"sample_sigmas": torch.tensor([50.0, 25.0, 0.0])},
    )

    assert changed is True
    assert group.strength == 3.0


def test_empty_weight_hook_schedule_uses_unit_multiplier() -> None:
    """Pin Comfy's unscheduled WeightHook multiplier contract."""

    hook = WeightHook(strength_model=-0.75)
    hook.hook_keyframe.initialize_timesteps(_Model())
    hook.hook_keyframe.reset()

    assert hook.strength == 1.0
    assert hook.strength_model == -0.75


@pytest.mark.parametrize(
    ("sigma", "expected"),
    [(75.01, 1.0), (75.0, 2.0), (74.99, 2.0)],
)
def test_weight_hook_boundary_is_inclusive(
    sigma: float,
    expected: float,
) -> None:
    """Pin exact comparison immediately around one converted boundary."""

    group = _group((0.0, 1.0, 0), (0.25, 2.0, 0))
    _initialize(group)

    group.prepare_current_keyframe(
        sigma,
        {"sample_sigmas": torch.tensor([100.0, sigma, 0.0])},
    )

    assert group.strength == expected


class _LinearSampling:
    """Convert authored percentages into recognizable descending sigmas."""

    @staticmethod
    def percent_to_sigma(percent: float) -> float:
        """Return one deterministic converted boundary."""

        return 100.0 * (1.0 - percent)


class _Model:
    """Expose the installed HookKeyframe initialization model boundary."""

    model_sampling = _LinearSampling()


def _group(*values: tuple[float, float, int]) -> HookKeyframeGroup:
    """Build one ordered installed Comfy keyframe group."""

    group = HookKeyframeGroup()
    for start_percent, strength, guarantee_steps in values:
        group.add(HookKeyframe(strength, start_percent, guarantee_steps))
    return group


def _initialize(group: HookKeyframeGroup) -> None:
    """Initialize and reset one installed Comfy schedule."""

    group.initialize_timesteps(_Model())
    group.reset()
