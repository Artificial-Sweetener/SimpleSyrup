# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose read-only CLIP hook-schedule state as managed benchmark evidence."""

from __future__ import annotations

import json
import math
from importlib import import_module
from typing import TYPE_CHECKING, Any

_comfy_api: Any = None
if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for the benchmark-only node."""

        pass

else:
    _comfy_api = import_module("comfy_api.latest")
    _ComfyNodeBase = _comfy_api.io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else _comfy_api.io


class SnapshotClipScheduleV3(_ComfyNodeBase):
    """Pass through one CLIP while publishing its hook registration state."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the CLIP passthrough and JSON evidence output."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.SnapshotClipSchedule",
            display_name="Benchmark Snapshot CLIP Schedule",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.Clip.Input("clip"),
                _comfy_io.String.Input("run_id"),
            ],
            outputs=[
                _comfy_io.Clip.Output("clip"),
                _comfy_io.String.Output("snapshot_json"),
            ],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, clip: object, run_id: str) -> Any:
        """Return the identical CLIP and one normalized schedule snapshot."""

        snapshot = snapshot_clip_schedule(clip, run_id=run_id)
        encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        return _comfy_io.NodeOutput(
            clip,
            encoded,
            ui={"clip_schedule_snapshot": [snapshot]},
        )


def snapshot_clip_schedule(clip: object, *, run_id: str) -> dict[str, object]:
    """Normalize one CLIP patcher's hook state without reading tensor contents."""

    if not isinstance(run_id, str) or not run_id:
        raise ValueError("CLIP schedule snapshot run ID must not be empty.")
    patcher = getattr(clip, "patcher", None)
    if patcher is None:
        raise TypeError("CLIP schedule snapshot requires a patcher.")
    patcher_model = getattr(patcher, "model", None)
    if patcher_model is None:
        raise TypeError("CLIP schedule snapshot patcher must expose its model.")
    text_encoder = getattr(clip, "cond_stage_model", None)
    if text_encoder is None:
        raise TypeError("CLIP schedule snapshot requires a text encoder.")
    hook_patches = getattr(patcher, "hook_patches", None)
    if not isinstance(hook_patches, dict):
        raise TypeError("CLIP patcher hook_patches must be a dictionary.")
    if any(not isinstance(targets, dict) for targets in hook_patches.values()):
        raise TypeError("CLIP patcher hook patch targets must be dictionaries.")
    forced_hooks = getattr(patcher, "forced_hooks", None)
    current_hooks = getattr(patcher, "current_hooks", None)
    return {
        "run_id": run_id,
        "clip_type": _type_name(clip),
        "patcher_type": _type_name(patcher),
        "text_encoder_type": _type_name(text_encoder),
        "text_encoder_is_patcher_model": text_encoder is patcher_model,
        "use_clip_schedule": _required_bool(clip, "use_clip_schedule"),
        "forced_hooks": _hook_group_snapshot(forced_hooks),
        "current_hooks": _hook_group_snapshot(current_hooks),
        "hook_patch_ref_count": len(hook_patches),
        "hook_patch_target_count": sum(
            len(targets) for targets in hook_patches.values()
        ),
        "hook_patch_refs": [
            _hook_ref(value, index) for index, value in enumerate(hook_patches)
        ],
    }


def _hook_group_snapshot(value: object | None) -> list[dict[str, object]]:
    """Normalize ordered hooks and their finite scalar schedule surface."""

    if value is None:
        return []
    hooks = getattr(value, "hooks", None)
    if not isinstance(hooks, list):
        raise TypeError("CLIP schedule hook group must expose an ordered hook list.")
    result: list[dict[str, object]] = []
    for index, hook in enumerate(hooks):
        keyframe_group = getattr(hook, "hook_keyframe", None)
        keyframes = getattr(keyframe_group, "keyframes", None)
        if not isinstance(keyframes, list):
            raise TypeError("CLIP schedule hook keyframes must be an ordered list.")
        result.append(
            {
                "order": index,
                "hook_type": _type_name(hook),
                "hook_ref": _hook_ref(getattr(hook, "hook_ref", None), index),
                "base_strength_model": _finite_number(
                    getattr(hook, "_strength_model", None),
                    field="model strength",
                ),
                "base_strength_clip": _finite_number(
                    getattr(hook, "_strength_clip", None),
                    field="CLIP strength",
                ),
                "effective_strength_clip": _finite_number(
                    getattr(hook, "strength_clip", None),
                    field="effective CLIP strength",
                ),
                "keyframes": [
                    {
                        "start_percent": _finite_number(
                            getattr(keyframe, "start_percent", None),
                            field="keyframe start",
                        ),
                        "strength": _finite_number(
                            getattr(keyframe, "strength", None),
                            field="keyframe strength",
                        ),
                        "guarantee_steps": _required_int(
                            keyframe,
                            "guarantee_steps",
                        ),
                    }
                    for keyframe in keyframes
                ],
            }
        )
    return result


def _required_bool(value: object, name: str) -> bool:
    """Return one required boolean attribute."""

    result = getattr(value, name, None)
    if not isinstance(result, bool):
        raise TypeError(f"CLIP {name} must be a boolean.")
    return result


def _required_int(value: object, name: str) -> int:
    """Return one required integer attribute without accepting booleans."""

    result = getattr(value, name, None)
    if isinstance(result, bool) or not isinstance(result, int):
        raise TypeError(f"CLIP schedule {name} must be an integer.")
    return result


def _finite_number(value: object, *, field: str) -> float:
    """Return one finite numeric schedule value without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"CLIP schedule {field} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"CLIP schedule {field} must be finite.")
    return result


def _hook_ref(value: object, index: int) -> str:
    """Preserve stable string refs and normalize native opaque refs."""

    return value if isinstance(value, str) else f"opaque-hook-ref-{index}"


def _type_name(value: object) -> str:
    """Return one stable module-qualified runtime type name."""

    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"
