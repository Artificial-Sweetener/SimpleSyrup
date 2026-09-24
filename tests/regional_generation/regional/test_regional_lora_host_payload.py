# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact snapshots of Comfy WeightHook payload state."""

from __future__ import annotations

from typing import Any, cast

import comfy.hooks
import pytest

from simple_syrup.runtime.regional_lora_host_payload import RegionalLoraHostPayload


def test_payload_preserves_unresolved_hook_weight_identity() -> None:
    """Retain raw source weights when Comfy has not initialized the hook."""

    raw_weights = {"diffusion_model.layer.lora_A.weight": object()}
    hook = comfy.hooks.create_hook_lora(raw_weights, 0.8, 0.0).get_type(
        comfy.hooks.EnumHookType.Weight
    )[0]

    payload = RegionalLoraHostPayload.from_hook(hook)

    assert payload.needs_resolution is True
    assert payload.raw_weights is raw_weights
    assert payload.model_weights is None
    assert payload.clip_weights is None
    assert payload.model_side_weights is raw_weights


def test_payload_preserves_initialized_model_and_clip_weight_identities() -> None:
    """Retain both exact initialized mappings without conflating their owners."""

    hook = comfy.hooks.create_hook_lora({}, 0.8, 0.6).get_type(
        comfy.hooks.EnumHookType.Weight
    )[0]
    model_weights = {"diffusion_model.layer.weight": object()}
    clip_weights = {"text_model.layer.weight": object()}
    hook.need_weight_init = False
    hook.weights = model_weights
    hook.weights_clip = clip_weights

    payload = RegionalLoraHostPayload.from_hook(hook)

    assert payload.needs_resolution is False
    assert payload.raw_weights is None
    assert payload.model_weights is model_weights
    assert payload.clip_weights is clip_weights
    assert payload.model_side_weights is model_weights


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ((True, None, None, None), "requires raw weights"),
        ((True, object(), object(), None), "cannot contain initialized"),
        ((False, object(), None, None), "cannot retain raw weights"),
    ],
)
def test_payload_rejects_ambiguous_host_state(
    payload: tuple[bool, object | None, object | None, object | None],
    message: str,
) -> None:
    """Reject snapshots that cannot represent a real WeightHook state."""

    with pytest.raises(ValueError, match=message):
        RegionalLoraHostPayload(*payload)


def test_payload_rejects_non_boolean_resolution_state() -> None:
    """Keep raw-versus-initialized state explicit and strictly typed."""

    with pytest.raises(TypeError, match="must be boolean"):
        RegionalLoraHostPayload(cast(Any, 1), object(), None, None)


def test_payload_factory_rejects_non_weight_hooks() -> None:
    """Prevent arbitrary host objects from entering the snapshot boundary."""

    with pytest.raises(TypeError, match="requires a WeightHook"):
        RegionalLoraHostPayload.from_hook(cast(Any, object()))
