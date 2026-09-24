# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify stable normalization of Comfy static and scheduled LoRA patches."""

from __future__ import annotations

from dataclasses import dataclass

from tools.attention_coupling_benchmark.comfy_probe.lora_patch_snapshot import (
    schedule_signature,
    snapshot_hook_patches,
    snapshot_static_patches,
)


@dataclass
class _Hook:
    """Provide the WeightHook values consumed by snapshot normalization."""

    hook_ref: str
    _strength_model: float
    strength: float

    @property
    def strength_model(self) -> float:
        """Return Comfy's effective model strength."""

        return self._strength_model * self.strength


@dataclass
class _HookGroup:
    """Expose ordered hooks through Comfy's group shape."""

    hooks: list[_Hook]


class _Model:
    """Expose only static and scheduled patch state."""

    def __init__(self) -> None:
        """Create consistent two-adapter patches on two targets."""

        patch_order = [
            (0.4, object(), 1.0, None, None),
            (0.6, object(), 1.0, None, None),
        ]
        self.patches = {
            "diffusion_model.a.weight": patch_order.copy(),
            "diffusion_model.b.weight": patch_order.copy(),
        }
        first = _Hook("first-ref", 0.4, 1.0)
        second = _Hook("second-ref", 0.6, 0.5)
        self.current_hooks = _HookGroup([first, second])
        self.hook_patches = {
            "first-ref": {"diffusion_model.a.weight": [object()]},
            "second-ref": {
                "diffusion_model.a.weight": [object()],
                "diffusion_model.b.weight": [object()],
            },
        }


def test_static_snapshot_records_targets_and_order() -> None:
    """Prove the declared identity order is consistent across every target."""

    snapshot = snapshot_static_patches(_Model(), ("adapter-a", "adapter-b"))

    assert snapshot["target_count"] == 2
    assert snapshot["entry_count_histogram"] == {"2": 2}
    assert snapshot["inconsistent_order_targets"] == []
    order = snapshot["canonical_patch_order"]
    assert isinstance(order, list)
    assert [entry["identity"] for entry in order] == ["adapter-a", "adapter-b"]
    assert [entry["strength_patch"] for entry in order] == [0.4, 0.6]


def test_hook_snapshot_preserves_order_targets_and_effective_strength() -> None:
    """Normalize registered hooks without exposing unstable hook UUIDs."""

    hooks = snapshot_hook_patches(_Model(), ("adapter-a", "adapter-b"))

    assert [hook["identity"] for hook in hooks] == ["adapter-a", "adapter-b"]
    assert [hook["target_count"] for hook in hooks] == [1, 2]
    assert schedule_signature(hooks) == (0.4, 0.3)
