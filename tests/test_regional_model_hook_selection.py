"""Prove authoritative model participation selection from Comfy HookGroups."""

from __future__ import annotations

from typing import Any

import comfy.hooks
import pytest

from simple_syrup.runtime.regional_model_hook_selection import (
    REGIONAL_MODEL_HOOK_SELECTOR,
)


def test_selector_preserves_only_nonzero_model_hooks_in_original_order() -> None:
    """Omit text-only hooks while retaining signed model participants exactly."""

    text_only = _hook(0.0, 0.75)
    positive = _hook(0.8, 0.0)
    negative = _hook(-0.4, 0.5)
    hooks = comfy.hooks.HookGroup.combine_all_hooks((text_only, positive, negative))
    assert hooks is not None
    source_hooks = tuple(hooks.hooks)

    selection = REGIONAL_MODEL_HOOK_SELECTOR.select(
        hooks,
        source_label="regional fixture",
    )

    assert selection.weight_hook_count == 3
    assert [item.hook_index for item in selection.model_hooks] == [1, 2]
    assert [item.model_strength for item in selection.model_hooks] == [0.8, -0.4]
    assert [item.hook for item in selection.model_hooks] == [
        source_hooks[1],
        source_hooks[2],
    ]
    assert tuple(hooks.hooks) == source_hooks


def test_selector_accepts_empty_and_text_only_groups() -> None:
    """Represent no model request without consulting CLIP weights or identities."""

    empty = comfy.hooks.HookGroup()
    text_only = _hook(-0.0, 1.0)

    assert (
        REGIONAL_MODEL_HOOK_SELECTOR.select(
            empty,
            source_label="empty fixture",
        ).model_hooks
        == ()
    )
    selection = REGIONAL_MODEL_HOOK_SELECTOR.select(
        text_only,
        source_label="text fixture",
    )
    assert selection.weight_hook_count == 1
    assert selection.model_hooks == ()


@pytest.mark.parametrize("value", [True, "1.0", float("nan"), float("inf")])
def test_selector_rejects_malformed_model_strength(value: object) -> None:
    """Fail closed before malformed host state controls model participation."""

    hooks = _hook(1.0, 0.0)
    hooks.get_type(comfy.hooks.EnumHookType.Weight)[0]._strength_model = value

    with pytest.raises(TypeError, match="model strength must be finite numeric"):
        REGIONAL_MODEL_HOOK_SELECTOR.select(
            hooks,
            source_label="malformed fixture",
        )


def test_selector_rejects_non_weight_hooks_and_non_groups() -> None:
    """Keep unsupported model behavior out of the WeightHook path."""

    unsupported = comfy.hooks.HookGroup()
    unsupported.add(
        comfy.hooks.Hook(hook_type=comfy.hooks.EnumHookType.TransformerOptions)
    )

    with pytest.raises(TypeError, match="unsupported hooks: Hook"):
        REGIONAL_MODEL_HOOK_SELECTOR.select(
            unsupported,
            source_label="unsupported fixture",
        )
    with pytest.raises(TypeError, match="must contain a Comfy HookGroup"):
        REGIONAL_MODEL_HOOK_SELECTOR.select(
            object(),
            source_label="invalid fixture",
        )


def _hook(model_strength: float, clip_strength: float) -> comfy.hooks.HookGroup:
    """Create one raw hook without requiring real tensor payloads."""

    return comfy.hooks.create_hook_lora(
        {"fixture.weight": Any},
        strength_model=model_strength,
        strength_clip=clip_strength,
    )
