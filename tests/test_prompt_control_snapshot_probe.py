"""Verify pre-sampling Prompt Control conditioning and HookGroup snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from tools.attention_coupling_benchmark.comfy_probe.prompt_control_snapshot import (
    SnapshotPromptControlV3,
    snapshot_conditioning,
    snapshot_hook_group,
)


@dataclass
class _Keyframe:
    """Provide native keyframe attributes to the dynamic probe boundary."""

    start_percent: float
    strength: float
    guarantee_steps: int = 1


@dataclass
class _Keyframes:
    """Expose ordered fake native keyframes."""

    keyframes: list[_Keyframe]


@dataclass
class _Hook:
    """Provide the WeightHook fields preserved by Prompt Control."""

    _strength_model: float = 0.75
    _strength_clip: float = 0.25
    hook_keyframe: _Keyframes = field(
        default_factory=lambda: _Keyframes([_Keyframe(0.0, 0.0), _Keyframe(0.25, 1.0)])
    )
    hook_ref: object = field(default_factory=object)
    hook_id: str | None = None
    hook_scope: object = None

    @property
    def strength_model(self) -> float:
        """Return the initial effective model strength."""

        return 0.0

    @property
    def strength_clip(self) -> float:
        """Return the initial effective CLIP strength."""

        return 0.0


@dataclass
class _Hooks:
    """Expose an ordered hook list."""

    hooks: list[_Hook]


def test_snapshot_preserves_entries_metadata_and_keyframes() -> None:
    """Record entry order, tensor identity, exact intervals, and hook schedule."""

    hooks = _Hooks([_Hook()])
    conditioning = [
        [
            torch.ones((1, 2, 3)),
            {
                "start_percent": 0.25,
                "end_percent": 0.75,
                "strength": 0.6,
                "hooks": hooks,
            },
        ]
    ]
    snapshot = snapshot_conditioning(conditioning, ("adapter-a",))
    hook_snapshot = snapshot_hook_group(hooks, ("adapter-a",))
    metadata = snapshot[0]["metadata"]
    assert isinstance(metadata, dict)
    hook_metadata = metadata["hooks"]
    assert isinstance(hook_metadata, dict)
    hook_entries = hook_metadata["hook_group"]
    assert isinstance(hook_entries, list)
    assert metadata["start_percent"] == 0.25
    assert hook_entries[0]["identity"] == "adapter-a"
    assert hook_snapshot[0]["keyframes"] == [
        {"start_percent": 0.0, "strength": 0.0, "guarantee_steps": 1},
        {"start_percent": 0.25, "strength": 1.0, "guarantee_steps": 1},
    ]


def test_snapshot_schema_is_dev_only_and_unknown_metadata_fails_closed() -> None:
    """Keep the probe isolated and reject unserialized metadata owners."""

    schema = SnapshotPromptControlV3.define_schema()
    assert schema.node_id == "SimpleSyrupBenchmark.SnapshotPromptControl"
    assert schema.is_output_node is True
    assert schema.is_dev_only is True
    with pytest.raises(TypeError, match="Unsupported conditioning metadata"):
        snapshot_conditioning([[torch.zeros(1), {"unknown": object()}]], ())
