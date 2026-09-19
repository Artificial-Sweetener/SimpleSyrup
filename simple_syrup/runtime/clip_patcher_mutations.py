# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own concrete mutations applied through ComfyUI's CLIP patcher API."""

from __future__ import annotations

from collections.abc import Callable
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


@dataclass(frozen=True)
class ClipCallableObjectPatchMutation:
    """Patch one callable text-encoder object on a derived CLIP patcher."""

    path: str
    replacement: Callable[..., object]

    def apply(self, clip: object) -> None:
        """Validate the path and collision state before installing the callback."""

        if (
            not isinstance(self.path, str)
            or not self.path
            or any(not segment for segment in self.path.split("."))
        ):
            raise ValueError("CLIP callable patch path must be a dotted path.")
        if not callable(self.replacement):
            raise TypeError("CLIP callable object replacement must be callable.")
        patcher = _required_attribute(clip, "patcher", value_name="CLIP")
        getter = getattr(patcher, "get_model_object", None)
        adder = getattr(patcher, "add_object_patch", None)
        object_patches = getattr(patcher, "object_patches", None)
        if (
            not callable(getter)
            or not callable(adder)
            or not isinstance(object_patches, dict)
        ):
            raise TypeError("CLIP patcher does not expose callable object patches.")
        if self.path in object_patches:
            raise ValueError(f"CLIP object path '{self.path}' already has a patch.")
        if not callable(getter(self.path)):
            raise TypeError(f"CLIP object path '{self.path}' must be callable.")
        adder(self.path, self.replacement)


@dataclass(frozen=True)
class ClipTokenizerMutation:
    """Replace the tokenizer on a derived CLIP after exact source validation."""

    expected_source: object
    replacement: object

    def apply(self, clip: object) -> None:
        """Install one tokenizer proxy only on the expected cloned source value."""

        if getattr(clip, "tokenizer", None) is not self.expected_source:
            raise ValueError("Derived CLIP tokenizer does not match its source.")
        cast(Any, clip).tokenizer = self.replacement


@dataclass(frozen=True)
class ClipBooleanOptionMutation:
    """Publish one collision-safe boolean option on a derived CLIP patcher."""

    key: str
    value: bool

    def apply(self, clip: object) -> None:
        """Set an approved ownership marker after validating the option mapping."""

        if self.key not in {"ppm_negpip", "simple_syrup_negpip"}:
            raise ValueError("Unsupported CLIP boolean option marker.")
        if not isinstance(self.value, bool):
            raise TypeError("CLIP option marker value must be boolean.")
        patcher = _required_attribute(clip, "patcher", value_name="CLIP")
        options = getattr(patcher, "model_options", None)
        if not isinstance(options, dict):
            raise TypeError("CLIP patcher model_options must be a dictionary.")
        if self.key in options:
            raise ValueError(f"CLIP option '{self.key}' is already present.")
        options[self.key] = self.value


def _required_attribute(value: object, name: str, *, value_name: str) -> object:
    """Return a required dynamic ComfyUI boundary attribute."""

    attribute = getattr(value, name, None)
    if attribute is None:
        raise TypeError(f"{value_name} does not expose {name}.")
    return attribute
