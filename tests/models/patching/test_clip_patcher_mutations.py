# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize concrete CLIP patcher mutation behavior."""

from __future__ import annotations

import pytest

from simple_syrup.runtime.clip_patcher_mutations import (
    ClipHookScheduleMutation,
    ClipLayerMutation,
)


class _RecordingHooks:
    """Provide lifecycle-safe hook cloning."""

    def __init__(self, cloned: object) -> None:
        """Store the clone result."""

        self.cloned = cloned

    def clone(self) -> object:
        """Return the configured clone."""

        return self.cloned


class _RecordingPatcher:
    """Record CLIP hook registration state."""

    def __init__(self) -> None:
        """Initialize empty registration state."""

        self.forced_hooks: object | None = None
        self.registrations: list[tuple[object, object]] = []

    def register_all_hook_patches(self, hooks: object, target: object) -> None:
        """Record one native hook registration."""

        self.registrations.append((hooks, target))


class _RecordingClip:
    """Expose the supported CLIP mutation surface."""

    def __init__(self) -> None:
        """Initialize layer and schedule state."""

        self.patcher = _RecordingPatcher()
        self.layer_index: int | None = None
        self.use_clip_schedule = False

    def clip_layer(self, layer_index: int) -> None:
        """Record the selected text-encoder layer."""

        self.layer_index = layer_index


def test_clip_layer_mutation_uses_exact_comfy_surface() -> None:
    """Select the configured layer through CLIP's public method."""

    clip = _RecordingClip()

    ClipLayerMutation(-2).apply(clip)

    assert clip.layer_index == -2


def test_clip_hook_schedule_mutation_clones_and_registers_hooks() -> None:
    """Install a cloned forced-hook set and register the original schedule."""

    clip = _RecordingClip()
    cloned_hooks = object()
    hooks = _RecordingHooks(cloned_hooks)
    target = object()

    ClipHookScheduleMutation(hooks, target).apply(clip)

    assert clip.patcher.forced_hooks is cloned_hooks
    assert clip.use_clip_schedule is True
    assert clip.patcher.registrations == [(hooks, target)]


def test_clip_layer_mutation_rejects_missing_layer_surface() -> None:
    """Fail closed when CLIP cannot select a text-encoder layer."""

    with pytest.raises(TypeError, match="CLIP does not support layer selection"):
        ClipLayerMutation(-2).apply(object())


@pytest.mark.parametrize(
    ("clip", "hooks", "message"),
    [
        (object(), _RecordingHooks(object()), "CLIP does not expose patcher"),
        (
            _RecordingClip(),
            object(),
            "HOOKS does not support lifecycle-safe cloning",
        ),
        (
            type("ClipWithoutRegistration", (), {"patcher": object()})(),
            _RecordingHooks(object()),
            "CLIP patcher does not support hook registration",
        ),
    ],
)
def test_clip_hook_schedule_rejects_missing_comfy_surface(
    clip: object,
    hooks: object,
    message: str,
) -> None:
    """Reject every incomplete native CLIP hook boundary."""

    with pytest.raises(TypeError, match=message):
        ClipHookScheduleMutation(hooks, object()).apply(clip)
