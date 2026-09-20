# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compose global-first conditioning hooks for conventional regional sampling."""

from __future__ import annotations

from typing import Any, TypeAlias

from comfy.hooks import HookGroup

from .regional_lora_conditioning_sources import conditioning_hook_groups

Conditioning: TypeAlias = list[list[Any]]


class GlobalFirstConditioningHookComposer:
    """Apply one global HookGroup to every regional conditioning model state."""

    def global_hooks(
        self,
        conditioning: Conditioning,
        *,
        source_label: str,
    ) -> HookGroup | None:
        """Return the single uniform HookGroup carried by global conditioning."""

        groups = conditioning_hook_groups(conditioning)
        if not groups:
            return None
        authority = groups[0]
        if any(group is not authority for group in groups[1:]):
            raise ValueError(
                f"{source_label} uses different HookGroups across conditioning "
                "entries. Keep one shared Prompt Control hook schedule on the "
                "global segment."
            )
        return authority

    def compose(
        self,
        conditioning: Conditioning,
        global_hooks: HookGroup | None,
        *,
        source_label: str,
        cache: dict[tuple[HookGroup, HookGroup], HookGroup],
    ) -> Conditioning:
        """Prepend global hooks to every local HookGroup without mutating inputs."""

        if global_hooks is None:
            return [[item[0], dict(item[1])] for item in conditioning]
        composed: Conditioning = []
        for item_index, item in enumerate(conditioning):
            metadata = dict(item[1])
            local_hooks = metadata.get("hooks")
            if local_hooks is None:
                metadata["hooks"] = global_hooks
            elif not isinstance(local_hooks, HookGroup):
                raise TypeError(
                    f"{source_label} item {item_index} hooks must be a Comfy HookGroup."
                )
            else:
                key = (global_hooks, local_hooks)
                combined = cache.get(key)
                if combined is None:
                    combined = global_hooks.clone_and_combine(local_hooks)
                    cache[key] = combined
                metadata["hooks"] = combined
            composed.append([item[0], metadata])
        return composed


GLOBAL_FIRST_CONDITIONING_HOOK_COMPOSER = GlobalFirstConditioningHookComposer()
