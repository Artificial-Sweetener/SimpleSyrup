# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove regional LoRA schedule resolution matches installed Comfy state."""

from __future__ import annotations

import pytest
import torch
from comfy.hooks import HookKeyframe, HookKeyframeGroup, WeightHook

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.runtime.regional_lora_schedule_resolution import (
    RegionalLoraScheduleSession,
)


def test_schedule_session_matches_independent_installed_weight_hooks() -> None:
    """Match order, boundaries, base strengths, repeats, and guarantees exactly."""

    first_hook = _hook(1.5, ((0.0, 0.0, 0), (0.25, 1.0, 0), (0.5, 0.5, 0)))
    second_hook = _hook(-2.0, ((0.0, 1.0, 2), (0.25, 0.0, 0)))
    hooks = (first_hook, second_hook)
    adapters = (
        _adapter(
            0, 1.5, ((0.0, 100.0, 0.0, 0), (0.25, 75.0, 1.0, 0), (0.5, 50.0, 0.5, 0))
        ),
        _adapter(1, -2.0, ((0.0, 100.0, 1.0, 2), (0.25, 75.0, 0.0, 0))),
    )
    options = {"sample_sigmas": torch.tensor([100.0, 75.0, 50.0, 25.0, 0.0])}
    session = RegionalLoraScheduleSession(adapters, maximum_sigma=100.0)

    for sigma in (100.0, 75.0, 75.0, 50.0, 25.0):
        for hook in hooks:
            hook.hook_keyframe.prepare_current_keyframe(sigma, options)
        resolved = session.resolve(sigma)
        assert resolved.schedule_multipliers == tuple(hook.strength for hook in hooks)
        assert resolved.effective_strengths == tuple(
            hook.strength_model for hook in hooks
        )


def test_schedule_session_matches_truncated_range_guarantee_behavior() -> None:
    """Drop guarantees whose current boundary precedes the sampled range."""

    hook = _hook(
        1.0,
        ((0.0, 1.0, 5), (0.25, 2.0, 0), (0.5, 3.0, 0)),
    )
    session = RegionalLoraScheduleSession(
        (
            _adapter(
                0,
                1.0,
                (
                    (0.0, 100.0, 1.0, 5),
                    (0.25, 75.0, 2.0, 0),
                    (0.5, 50.0, 3.0, 0),
                ),
            ),
        ),
        maximum_sigma=50.0,
    )
    options = {"sample_sigmas": torch.tensor([50.0, 25.0, 0.0])}

    hook.hook_keyframe.prepare_current_keyframe(50.0, options)
    resolved = session.resolve(50.0)

    assert resolved.schedule_multipliers == (hook.strength,) == (3.0,)


@pytest.mark.parametrize(
    ("maximum", "sigma", "error", "message"),
    [
        (True, 1.0, TypeError, "maximum sigma"),
        (float("nan"), 1.0, ValueError, "maximum sigma"),
        (1.0, True, TypeError, "current sigma"),
        (1.0, float("inf"), ValueError, "current sigma"),
    ],
)
def test_schedule_session_rejects_malformed_sigma_state(
    maximum: object,
    sigma: object,
    error: type[Exception],
    message: str,
) -> None:
    """Fail closed on invalid sampling-session metadata."""

    if isinstance(maximum, bool) or maximum != maximum:
        with pytest.raises(error, match=message):
            RegionalLoraScheduleSession((), maximum_sigma=maximum)  # type: ignore[arg-type]
        return
    session = RegionalLoraScheduleSession((), maximum_sigma=maximum)  # type: ignore[arg-type]
    with pytest.raises(error, match=message):
        session.resolve(sigma)  # type: ignore[arg-type]


class _LinearSampling:
    """Convert percentages into the test plan's exact sigmas."""

    @staticmethod
    def percent_to_sigma(percent: float) -> float:
        """Return a descending converted boundary."""

        return 100.0 * (1.0 - percent)


class _Model:
    """Expose installed HookKeyframe initialization state."""

    model_sampling = _LinearSampling()


def _hook(
    model_strength: float,
    values: tuple[tuple[float, float, int], ...],
) -> WeightHook:
    """Build and initialize one installed Comfy WeightHook schedule."""

    hook = WeightHook(strength_model=model_strength)
    group = HookKeyframeGroup()
    for start, strength, guarantee in values:
        group.add(HookKeyframe(strength, start, guarantee))
    group.initialize_timesteps(_Model())
    group.reset()
    hook.hook_keyframe = group
    return hook


def _adapter(
    index: int,
    model_strength: float,
    values: tuple[tuple[float, float, float, int], ...],
) -> RegionalLoraAdapterPlan:
    """Build one exact converted regional adapter schedule."""

    return RegionalLoraAdapterPlan(
        RegionalLoraAdapterIdentity(f"adapter-{index}"),
        index,
        index,
        RegionalLoraBranch.POSITIVE,
        model_strength,
        tuple(RegionalLoraScheduleBoundary(*value) for value in values),
    )
