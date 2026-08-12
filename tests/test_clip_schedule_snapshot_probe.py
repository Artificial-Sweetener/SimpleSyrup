"""Verify read-only benchmark snapshots of CLIP hook schedule state."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from tools.attention_coupling_benchmark.comfy_probe.clip_schedule_snapshot import (
    SnapshotClipScheduleV3,
    snapshot_clip_schedule,
)


@dataclass
class _Keyframe:
    """Expose the scalar native keyframe surface."""

    start_percent: float = 0.0
    strength: float = 1.0
    guarantee_steps: int = 1


@dataclass
class _Keyframes:
    """Expose ordered keyframes."""

    keyframes: list[_Keyframe] = field(default_factory=lambda: [_Keyframe()])


@dataclass
class _Hook:
    """Expose one registered text-encoder hook."""

    hook_ref: str = "fixture"
    _strength_model: float = 0.0
    _strength_clip: float = 0.75
    strength_clip: float = 0.75
    hook_keyframe: _Keyframes = field(default_factory=_Keyframes)


@dataclass
class _Hooks:
    """Expose an ordered hook collection."""

    hooks: list[_Hook] = field(default_factory=lambda: [_Hook()])


@dataclass
class _Patcher:
    """Expose the read-only patcher state used by the probe."""

    model: object = field(default_factory=object)
    forced_hooks: _Hooks = field(default_factory=_Hooks)
    current_hooks: _Hooks | None = None
    hook_patches: dict[object, dict[str, object]] = field(
        default_factory=lambda: {
            "fixture": {"target.a": object(), "target.b": object()}
        }
    )


@dataclass
class _Clip:
    """Expose one prepared CLIP surface."""

    patcher: _Patcher = field(default_factory=_Patcher)
    cond_stage_model: object | None = None
    use_clip_schedule: bool = True

    def __post_init__(self) -> None:
        """Default the text encoder to the patcher-owned model."""

        if self.cond_stage_model is None:
            self.cond_stage_model = self.patcher.model


def test_snapshot_records_exact_schedule_state_without_mutation() -> None:
    """Preserve hook order, strengths, keyframes, and patch cardinality."""

    clip = _Clip()
    snapshot = snapshot_clip_schedule(clip, run_id="run")

    assert snapshot["run_id"] == "run"
    assert snapshot["use_clip_schedule"] is True
    assert snapshot["hook_patch_ref_count"] == 1
    assert snapshot["hook_patch_target_count"] == 2
    assert snapshot["hook_patch_refs"] == ["fixture"]
    assert snapshot["text_encoder_is_patcher_model"] is True
    forced = snapshot["forced_hooks"]
    assert isinstance(forced, list)
    assert forced[0]["base_strength_clip"] == 0.75
    assert forced[0]["keyframes"] == [
        {"start_percent": 0.0, "strength": 1.0, "guarantee_steps": 1}
    ]
    assert clip.patcher.forced_hooks.hooks[0].hook_ref == "fixture"


def test_snapshot_observes_misaligned_text_encoder_without_repairing_it() -> None:
    """Report host CLIP/patcher model divergence without mutating either owner."""

    source_encoder = object()
    clip = _Clip(cond_stage_model=source_encoder)

    snapshot = snapshot_clip_schedule(clip, run_id="misaligned")

    assert snapshot["text_encoder_is_patcher_model"] is False
    assert clip.cond_stage_model is source_encoder


def test_snapshot_rejects_missing_or_malformed_patcher_state() -> None:
    """Fail closed on absent patchers and non-mapping registered targets."""

    with pytest.raises(TypeError, match="requires a patcher"):
        snapshot_clip_schedule(object(), run_id="run")
    clip = _Clip()
    clip.patcher.hook_patches = {"fixture": object()}  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="targets must be dictionaries"):
        snapshot_clip_schedule(clip, run_id="run")


def test_snapshot_node_is_output_only_and_dev_only() -> None:
    """Keep the CLIP observer isolated from product node exports."""

    schema = SnapshotClipScheduleV3.define_schema()
    assert schema.node_id == "SimpleSyrupBenchmark.SnapshotClipSchedule"
    assert schema.is_output_node is True
    assert schema.is_dev_only is True
