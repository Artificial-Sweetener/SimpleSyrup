# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Assign explicit stable identities to cloned regional LoRA hooks."""

from __future__ import annotations

import json

from comfy.hooks import EnumHookType, HookGroup, WeightHook

from .patcher_lifecycle import PATCHER_LIFECYCLE


def label_regional_lora_hooks(hooks: object, identities_json: str) -> HookGroup:
    """Clone and label ordered WeightHooks without changing schedules or weights."""

    if not isinstance(hooks, HookGroup):
        raise TypeError("Regional LoRA identity labeling requires a Comfy HookGroup.")
    identities = _decode_identities(identities_json)
    labeled = PATCHER_LIFECYCLE.clone_hooks(
        hooks,
        operation="SimpleSyrup regional LoRA identity labeling",
    )
    weight_hooks = labeled.get_type(EnumHookType.Weight)
    if len(weight_hooks) != len(labeled.hooks):
        raise TypeError("Regional LoRA identity labeling supports WeightHooks only.")
    if len(weight_hooks) != len(identities):
        raise ValueError(
            "Regional LoRA identity count must match the ordered WeightHook count."
        )
    for hook, identity in zip(weight_hooks, identities, strict=True):
        if not isinstance(hook, WeightHook):
            raise TypeError("Regional LoRA identity labeling requires WeightHooks.")
        hook.hook_ref = identity
    return labeled


def _decode_identities(value: str) -> tuple[str, ...]:
    """Decode a unique ordered JSON string array."""

    if not isinstance(value, str):
        raise TypeError("Regional LoRA identities must be encoded as JSON text.")
    try:
        decoded: object = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("Regional LoRA identities must contain valid JSON.") from error
    if not isinstance(decoded, list) or not decoded:
        raise ValueError("Regional LoRA identities must be a non-empty JSON array.")
    if any(not isinstance(item, str) or not item.strip() for item in decoded):
        raise ValueError("Regional LoRA identities must contain non-empty strings.")
    identities = tuple(decoded)
    if len(set(identities)) != len(identities):
        raise ValueError("Regional LoRA identities must be unique.")
    return identities
