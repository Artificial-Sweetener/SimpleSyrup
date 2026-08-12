# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own concrete mutations applied through ComfyUI's CLIP patcher API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class ClipLayerMutation:
    """Select the text-encoder layer used by a derived CLIP value."""

    layer_index: int

    def apply(self, clip: object) -> None:
        """Apply the configured CLIP layer index."""

        select_layer = getattr(clip, "clip_layer", None)
        if not callable(select_layer):
            raise TypeError("CLIP does not support layer selection.")
        select_layer(self.layer_index)


@dataclass(frozen=True)
class ClipHookScheduleMutation:
    """Install a native ComfyUI hook schedule on a derived CLIP value."""

    hooks: object
    target: object

    def apply(self, clip: object) -> None:
        """Clone and register hooks through the derived CLIP patcher."""

        patcher = _required_attribute(clip, "patcher", value_name="CLIP")
        clone_hooks = getattr(self.hooks, "clone", None)
        if not callable(clone_hooks):
            raise TypeError("HOOKS does not support lifecycle-safe cloning.")
        register_hooks = getattr(patcher, "register_all_hook_patches", None)
        if not callable(register_hooks):
            raise TypeError("CLIP patcher does not support hook registration.")

        patcher_boundary = cast(Any, patcher)
        clip_boundary = cast(Any, clip)
        patcher_boundary.forced_hooks = clone_hooks()
        clip_boundary.use_clip_schedule = True
        register_hooks(self.hooks, self.target)


def _required_attribute(value: object, name: str, *, value_name: str) -> object:
    """Return a required dynamic ComfyUI boundary attribute."""

    attribute = getattr(value, name, None)
    if attribute is None:
        raise TypeError(f"{value_name} does not expose {name}.")
    return attribute
