"""Prove immutable regional LoRA planning from ordered Comfy HookGroups."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import comfy.hooks
import pytest

from simple_syrup.domain.regional_lora_plan import (
    EMPTY_REGIONAL_LORA_PLAN,
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraPlan,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.runtime.regional_lora_plan_adapter import (
    REGIONAL_LORA_PLAN_ADAPTER,
    RegionalLoraHookSource,
)

PINNED_ADAPTER_A_IDENTITY = (
    "<MODEL_ROOT>\\Loras\\Anima\\style\\adapter-a.safetensors"
)


def test_hook_adapter_preserves_identity_order_ownership_strength_and_schedules() -> (
    None
):
    """Copy all ordered regional adapter semantics without changing source hooks."""

    positive_hooks = comfy.hooks.create_hook_lora(
        {"first.lora_A.weight": object()},
        strength_model=-0.75,
        strength_clip=0.0,
    ).clone_and_combine(
        comfy.hooks.create_hook_lora(
            {"second.lora_A.weight": object()},
            strength_model=0.0,
            strength_clip=0.0,
        )
    )
    positive_weight_hooks = positive_hooks.get_type(comfy.hooks.EnumHookType.Weight)
    positive_schedule = comfy.hooks.HookKeyframeGroup()
    positive_schedule.add(
        comfy.hooks.HookKeyframe(
            strength=0.25,
            start_percent=0.0,
            guarantee_steps=2,
        )
    )
    positive_schedule.add(
        comfy.hooks.HookKeyframe(
            strength=-1.5,
            start_percent=0.4,
            guarantee_steps=0,
        )
    )
    positive_weight_hooks[0].hook_keyframe = positive_schedule

    negative_hooks = comfy.hooks.create_hook_lora(
        {"negative.lora_A.weight": object()},
        strength_model=1.25,
        strength_clip=0.0,
    )
    negative_weight_hook = negative_hooks.get_type(comfy.hooks.EnumHookType.Weight)[0]
    negative_schedule = comfy.hooks.HookKeyframeGroup()
    negative_schedule.add(
        comfy.hooks.HookKeyframe(
            strength=1.0,
            start_percent=0.2,
            guarantee_steps=1,
        )
    )
    negative_schedule.add(
        comfy.hooks.HookKeyframe(
            strength=0.0,
            start_percent=1.0,
            guarantee_steps=3,
        )
    )
    negative_weight_hook.hook_keyframe = negative_schedule
    positive_snapshot = _hook_snapshot(positive_hooks)
    negative_snapshot = _hook_snapshot(negative_hooks)
    adapter_a_identity = RegionalLoraAdapterIdentity(PINNED_ADAPTER_A_IDENTITY)

    adaptation = REGIONAL_LORA_PLAN_ADAPTER.adapt(
        (
            RegionalLoraHookSource(
                region_index=2,
                branch=RegionalLoraBranch.POSITIVE,
                hooks=positive_hooks,
                adapter_identities=(adapter_a_identity,),
            ),
            RegionalLoraHookSource(
                region_index=5,
                branch=RegionalLoraBranch.NEGATIVE,
                hooks=negative_hooks,
                adapter_identities=(adapter_a_identity,),
            ),
        ),
        model=_Model(),
    )
    plan = adaptation.plan

    assert [adapter.composition_index for adapter in plan.adapters] == [0, 1]
    assert [adapter.adapter_identity.value for adapter in plan.adapters] == [
        PINNED_ADAPTER_A_IDENTITY,
        PINNED_ADAPTER_A_IDENTITY,
    ]
    assert [adapter.region_index for adapter in plan.adapters] == [2, 5]
    assert [adapter.branch for adapter in plan.adapters] == [
        RegionalLoraBranch.POSITIVE,
        RegionalLoraBranch.NEGATIVE,
    ]
    assert [adapter.model_strength for adapter in plan.adapters] == [-0.75, 1.25]
    assert plan.adapters[0].schedule == (
        RegionalLoraScheduleBoundary(0.0, 100.0, 0.25, 2),
        RegionalLoraScheduleBoundary(0.4, 60.0, -1.5, 0),
    )
    assert plan.adapters[1].schedule == (
        RegionalLoraScheduleBoundary(0.2, 80.0, 1.0, 1),
        RegionalLoraScheduleBoundary(1.0, 0.0, 0.0, 3),
    )
    assert _hook_snapshot(positive_hooks) == positive_snapshot
    assert _hook_snapshot(negative_hooks) == negative_snapshot
    assert adaptation.adapter_weights[0] is positive_weight_hooks[0].weights
    assert adaptation.adapter_weights[1] is negative_weight_hook.weights


def test_hook_adapter_accepts_empty_source_sequence() -> None:
    """Represent the absence of regional model LoRAs with one canonical value."""

    assert (
        REGIONAL_LORA_PLAN_ADAPTER.adapt((), model=_Model()).plan
        is EMPTY_REGIONAL_LORA_PLAN
    )


def test_regional_lora_plan_values_are_frozen() -> None:
    """Prevent mutation of adapter identity, schedules, entries, and plan order."""

    identity = RegionalLoraAdapterIdentity("adapter.safetensors")
    boundary = RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0)
    adapter = RegionalLoraAdapterPlan(
        adapter_identity=identity,
        composition_index=0,
        region_index=0,
        branch=RegionalLoraBranch.POSITIVE,
        model_strength=1.0,
        schedule=(boundary,),
    )
    plan = RegionalLoraPlan((adapter,))

    with pytest.raises(FrozenInstanceError):
        identity.value = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        boundary.strength_multiplier = 0.5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        adapter.region_index = 3  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.adapters = ()  # type: ignore[misc]


@pytest.mark.parametrize("value", ["", "   ", cast(Any, None)])
def test_adapter_identity_rejects_empty_values(value: str) -> None:
    """Reject missing stable artifact identities."""

    with pytest.raises(ValueError, match="non-empty string"):
        RegionalLoraAdapterIdentity(value)


@pytest.mark.parametrize(
    ("start", "sigma", "strength", "guarantee", "error", "message"),
    [
        (-0.1, 100.0, 1.0, 0, ValueError, "in \\[0, 1\\]"),
        (1.1, 0.0, 1.0, 0, ValueError, "in \\[0, 1\\]"),
        (float("nan"), 100.0, 1.0, 0, TypeError, "finite float"),
        (0.0, float("nan"), 1.0, 0, TypeError, "finite float"),
        (0.0, -1.0, 1.0, 0, ValueError, "non-negative"),
        (0.0, 100.0, float("inf"), 0, TypeError, "finite float"),
        (0.0, 100.0, 1.0, -1, ValueError, "non-negative"),
        (0.0, 100.0, 1.0, cast(Any, True), TypeError, "integer"),
    ],
)
def test_schedule_boundary_rejects_malformed_fields(
    start: float,
    sigma: float,
    strength: float,
    guarantee: int,
    error: type[Exception],
    message: str,
) -> None:
    """Fail closed on every malformed schedule-boundary field."""

    with pytest.raises(error, match=message):
        RegionalLoraScheduleBoundary(start, sigma, strength, guarantee)


def test_adapter_plan_rejects_unordered_schedule_and_plan_indices() -> None:
    """Reject ambiguous schedule and global composition order."""

    identity = RegionalLoraAdapterIdentity("adapter.safetensors")
    later = RegionalLoraScheduleBoundary(0.8, 20.0, 1.0, 0)
    earlier = RegionalLoraScheduleBoundary(0.2, 80.0, 1.0, 0)

    with pytest.raises(ValueError, match="boundaries must be ordered"):
        RegionalLoraAdapterPlan(
            identity,
            0,
            0,
            RegionalLoraBranch.POSITIVE,
            1.0,
            (later, earlier),
        )
    valid = RegionalLoraAdapterPlan(
        identity,
        1,
        0,
        RegionalLoraBranch.POSITIVE,
        1.0,
        (earlier,),
    )
    with pytest.raises(ValueError, match="contiguous and ordered"):
        RegionalLoraPlan((valid,))

    with pytest.raises(ValueError, match="converted schedule boundaries"):
        RegionalLoraAdapterPlan(
            identity,
            0,
            0,
            RegionalLoraBranch.POSITIVE,
            1.0,
            (
                RegionalLoraScheduleBoundary(0.2, 20.0, 1.0, 0),
                RegionalLoraScheduleBoundary(0.8, 80.0, 1.0, 0),
            ),
        )


def test_hook_adapter_rejects_identity_count_mismatch() -> None:
    """Require an explicit stable identity for every ordered WeightHook."""

    hooks = comfy.hooks.create_hook_lora({}, 1.0, 0.0)
    source = RegionalLoraHookSource(
        0,
        RegionalLoraBranch.POSITIVE,
        hooks,
        (),
    )

    with pytest.raises(ValueError, match="0 adapter identities for 1"):
        REGIONAL_LORA_PLAN_ADAPTER.adapt((source,), model=_Model())


def test_hook_adapter_rejects_non_weight_hooks_and_non_hook_groups() -> None:
    """Reject HookGroup behaviors that regional model-LoRA planning cannot own."""

    unsupported_group = comfy.hooks.HookGroup()
    unsupported_group.add(
        comfy.hooks.Hook(hook_type=comfy.hooks.EnumHookType.TransformerOptions)
    )
    unsupported_source = RegionalLoraHookSource(
        0,
        RegionalLoraBranch.POSITIVE,
        unsupported_group,
        (),
    )
    invalid_source = RegionalLoraHookSource(
        0,
        RegionalLoraBranch.POSITIVE,
        object(),
        (),
    )

    with pytest.raises(TypeError, match="unsupported hooks: Hook"):
        REGIONAL_LORA_PLAN_ADAPTER.adapt((unsupported_source,), model=_Model())
    with pytest.raises(TypeError, match="must contain a Comfy HookGroup"):
        REGIONAL_LORA_PLAN_ADAPTER.adapt((invalid_source,), model=_Model())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("_strength_model", float("nan"), "model strength"),
        ("hook_keyframe", object(), "HookKeyframeGroup"),
    ],
)
def test_hook_adapter_rejects_malformed_weight_hook_state(
    field: str,
    value: object,
    message: str,
) -> None:
    """Fail closed when installed WeightHook state cannot be preserved exactly."""

    hooks = comfy.hooks.create_hook_lora({}, 1.0, 0.0)
    hook = hooks.get_type(comfy.hooks.EnumHookType.Weight)[0]
    setattr(hook, field, value)
    source = RegionalLoraHookSource(
        0,
        RegionalLoraBranch.POSITIVE,
        hooks,
        (RegionalLoraAdapterIdentity("adapter.safetensors"),),
    )

    with pytest.raises(TypeError, match=message):
        REGIONAL_LORA_PLAN_ADAPTER.adapt((source,), model=_Model())


class _LinearSampling:
    """Convert authored percentages into deterministic plan sigmas."""

    @staticmethod
    def percent_to_sigma(percent: float) -> float:
        """Return a descending converted boundary."""

        return 100.0 * (1.0 - percent)


class _Model:
    """Expose the regional plan adapter's model-sampling boundary."""

    model_sampling = _LinearSampling()


def _hook_snapshot(hooks: comfy.hooks.HookGroup) -> tuple[tuple[object, ...], ...]:
    """Return identity and semantic fields used to prove source immutability."""

    return tuple(
        (
            hook,
            hook.hook_ref,
            hook._strength_model,
            hook.hook_keyframe,
            tuple(
                (
                    keyframe,
                    keyframe.start_percent,
                    keyframe.strength,
                    keyframe.guarantee_steps,
                )
                for keyframe in hook.hook_keyframe.keyframes
            ),
        )
        for hook in hooks.get_type(comfy.hooks.EnumHookType.Weight)
    )
