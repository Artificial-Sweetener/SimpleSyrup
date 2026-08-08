# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare regional LoRA hooks without needlessly cloning text encoders."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

from .patcher_lifecycle import PATCHER_LIFECYCLE, ClipHookScheduleMutation

LOGGER = logging.getLogger(__name__)


def prepare_regional_lora_clip(clip: Any, hooks: object) -> tuple[Any, object]:
    """Prepare CLIP only when at least one hook contains CLIP-compatible weights."""

    comfy_hooks = import_module("comfy.hooks")
    if not isinstance(hooks, comfy_hooks.HookGroup):
        raise TypeError("regional LoRA hooks must be a Comfy HOOKS value.")
    matching_hook_count = _count_clip_patch_hooks(clip, hooks, comfy_hooks)
    if matching_hook_count == 0:
        LOGGER.debug(
            "Regional LoRA hook preparation preserved the original CLIP because "
            "the hook group contains no matching text-encoder patches."
        )
        return clip, hooks

    LOGGER.debug(
        "Regional LoRA hook preparation created a scheduled CLIP for %d "
        "text-encoder hook group entries.",
        matching_hook_count,
    )
    prepared_clip = PATCHER_LIFECYCLE.derive_clip(
        clip,
        (
            ClipHookScheduleMutation(
                hooks=hooks,
                target=comfy_hooks.create_target_dict(
                    comfy_hooks.EnumWeightTarget.Clip
                ),
            ),
        ),
        operation="SimpleSyrup regional LoRA CLIP preparation",
        disable_dynamic=True,
    )
    return prepared_clip, hooks


def _count_clip_patch_hooks(clip: Any, hooks: Any, comfy_hooks: Any) -> int:
    """Count enabled weight hooks that resolve to at least one CLIP model key."""

    comfy_lora = import_module("comfy.lora")
    key_map = comfy_lora.model_lora_keys_clip(clip.cond_stage_model, {})
    matching_hook_count = 0
    for hook in hooks.get_type(comfy_hooks.EnumHookType.Weight):
        if hook._strength_clip == 0.0:
            continue
        if hook.need_weight_init:
            loaded = comfy_lora.load_lora(hook.weights, key_map, log_missing=False)
        else:
            loaded = hook.weights_clip
        if loaded:
            matching_hook_count += 1
    return matching_hook_count
