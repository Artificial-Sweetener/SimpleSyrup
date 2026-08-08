# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for regional LoRA CLIP preparation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import comfy.hooks
import comfy.lora
import pytest

from simple_syrup.runtime.regional_lora_hooks import prepare_regional_lora_clip


class _FakeClip:
    """Record whether regional preparation clones the text encoder."""

    def __init__(self, parent_patcher: object | None = None) -> None:
        """Create a clip and its minimal patcher collaboration."""

        self.cond_stage_model = object()
        self.registrations: list[tuple[Any, Any]] = []
        self.patcher = SimpleNamespace(
            forced_hooks=None,
            parent=parent_patcher,
            register_all_hook_patches=self._register_hooks,
        )
        self.use_clip_schedule = False
        self.clone_calls: list[bool] = []

    def clone(self, disable_dynamic: bool = False) -> _FakeClip:
        """Return a distinct clip while recording clone policy."""

        self.clone_calls.append(disable_dynamic)
        clone = _FakeClip(parent_patcher=self.patcher)
        clone.clone_calls = self.clone_calls
        clone.registrations = self.registrations
        clone.patcher.register_all_hook_patches = clone._register_hooks
        return clone

    def _register_hooks(self, hooks: Any, target: Any) -> None:
        """Record native hook registration on the prepared clip."""

        self.registrations.append((hooks, target))


def test_model_only_hooks_preserve_original_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A diffusion-only LoRA never clones or patches the text encoder."""

    monkeypatch.setattr(comfy.lora, "model_lora_keys_clip", lambda model, keys: keys)
    monkeypatch.setattr(
        comfy.lora,
        "load_lora",
        lambda weights, key_map, log_missing: {},
    )
    hooks = comfy.hooks.create_hook_lora(
        {"diffusion_model.layer.lora_A.weight": object()},
        strength_model=1.0,
        strength_clip=1.0,
    )
    clip = _FakeClip()

    prepared_clip, prepared_hooks = prepare_regional_lora_clip(clip, hooks)

    assert prepared_clip is clip
    assert prepared_hooks is hooks
    assert clip.clone_calls == []


def test_clip_hooks_prepare_a_scheduled_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A LoRA with matching text-encoder weights retains native CLIP hooks."""

    monkeypatch.setattr(
        comfy.lora,
        "model_lora_keys_clip",
        lambda model, keys: {"clip_key": "source_key"},
    )
    monkeypatch.setattr(
        comfy.lora,
        "load_lora",
        lambda weights, key_map, log_missing: {"clip.weight": object()},
    )
    hooks = cast(
        Any,
        comfy.hooks.create_hook_lora(
            {"clip.source.lora_A.weight": object()},
            strength_model=1.0,
            strength_clip=0.75,
        ),
    )
    clip = _FakeClip()
    prepared_clip, prepared_hooks = prepare_regional_lora_clip(clip, hooks)

    assert prepared_clip is not clip
    assert prepared_hooks is hooks
    assert clip.clone_calls == [True]
    assert prepared_clip.use_clip_schedule is True
    assert prepared_clip.patcher.forced_hooks is not hooks
    forced_hooks = cast(Any, prepared_clip.patcher.forced_hooks)
    source_hooks = cast(Any, hooks)
    forced_hook = cast(Any, forced_hooks.hooks[0])
    source_hook = cast(Any, source_hooks.hooks[0])
    assert forced_hook.hook_ref is source_hook.hook_ref
    assert len(clip.registrations) == 1
    assert clip.registrations[0][0] is hooks


def test_zero_strength_clip_hooks_preserve_original_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled CLIP strength avoids matching and cloning the text encoder."""

    monkeypatch.setattr(
        comfy.lora,
        "model_lora_keys_clip",
        lambda model, keys: {"clip_key": "source_key"},
    )

    def fail_load_lora(*args: object, **kwargs: object) -> object:
        """Fail if disabled CLIP hooks attempt to resolve weights."""

        raise AssertionError("disabled CLIP hook should not load weights")

    monkeypatch.setattr(comfy.lora, "load_lora", fail_load_lora)
    hooks = comfy.hooks.create_hook_lora(
        {"clip.source.lora_A.weight": object()},
        strength_model=1.0,
        strength_clip=0.0,
    )
    clip = _FakeClip()

    prepared_clip, _ = prepare_regional_lora_clip(clip, hooks)

    assert prepared_clip is clip
    assert clip.clone_calls == []


def test_prepare_regional_lora_clip_rejects_non_hook_values() -> None:
    """Invalid graph values fail before CLIP or model registration."""

    with pytest.raises(TypeError, match="Comfy HOOKS"):
        prepare_regional_lora_clip(_FakeClip(), object())
