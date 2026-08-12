"""Verify clone-local stable identity labeling for regional LoRA hooks."""

from __future__ import annotations

import json

import comfy.hooks
import pytest

from simple_syrup.runtime.regional_lora_hook_identity import (
    label_regional_lora_hooks,
)


def test_labels_cloned_hooks_without_changing_weights_or_keyframes() -> None:
    """Preserve exact lazy schedules and payloads while assigning identities."""

    first = _hook(0.4, ((0.0, 1.0), (0.25, 0.0), (0.25, 1.0), (0.75, 0.0)))
    second = _hook(0.6, ((0.0, 0.0), (0.25, 1.0), (0.75, 0.0), (0.75, 1.0)))
    source = first.clone_and_combine(second)
    source_hooks = source.get_type(comfy.hooks.EnumHookType.Weight)
    source_refs = tuple(hook.hook_ref for hook in source_hooks)

    labeled = label_regional_lora_hooks(
        source,
        json.dumps(["adapter-a", "adapter-b"]),
    )
    labeled_hooks = labeled.get_type(comfy.hooks.EnumHookType.Weight)

    assert labeled is not source
    assert [hook.hook_ref for hook in labeled_hooks] == ["adapter-a", "adapter-b"]
    assert tuple(hook.hook_ref for hook in source_hooks) == source_refs
    assert [hook.weights for hook in labeled_hooks] == [
        hook.weights for hook in source_hooks
    ]
    assert [_keyframes(hook) for hook in labeled_hooks] == [
        _keyframes(hook) for hook in source_hooks
    ]


@pytest.mark.parametrize(
    ("identities", "message"),
    [
        ([], "non-empty"),
        (["same", "same"], "unique"),
        (["one"], "count must match"),
    ],
)
def test_rejects_invalid_or_misaligned_identity_lists(
    identities: list[str],
    message: str,
) -> None:
    """Require one unique non-empty identity for every exact WeightHook."""

    hooks = _hook(1.0, ((0.0, 1.0),)).clone_and_combine(_hook(0.5, ((0.0, 1.0),)))

    with pytest.raises(ValueError, match=message):
        label_regional_lora_hooks(hooks, json.dumps(identities))


def _hook(
    strength: float,
    keyframes: tuple[tuple[float, float], ...],
) -> comfy.hooks.HookGroup:
    """Return one weight hook with an exact ordered native keyframe chain."""

    hooks = comfy.hooks.create_hook_lora(
        {"diffusion_model.layer.lora_A.weight": object()},
        strength_model=strength,
        strength_clip=strength,
    )
    schedule = comfy.hooks.HookKeyframeGroup()
    for start, value in keyframes:
        schedule.add(
            comfy.hooks.HookKeyframe(
                strength=value,
                start_percent=start,
                guarantee_steps=1,
            )
        )
    hooks.set_keyframes_on_hooks(schedule)
    return hooks


def _keyframes(hook: comfy.hooks.WeightHook) -> tuple[tuple[float, float, int], ...]:
    """Return immutable scalar keyframe evidence."""

    return tuple(
        (item.start_percent, item.strength, item.guarantee_steps)
        for item in hook.hook_keyframe.keyframes
    )
