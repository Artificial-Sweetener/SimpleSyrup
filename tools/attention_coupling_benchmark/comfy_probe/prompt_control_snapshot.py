"""Expose Prompt Control conditioning and HookGroup metadata as JSON evidence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any

import torch

from .tensor_snapshot import snapshot_tensor

_comfy_api: Any = None
if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for the benchmark-only node."""

        pass

else:
    _comfy_api = import_module("comfy_api.latest")
    _ComfyNodeBase = _comfy_api.io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else _comfy_api.io


class SnapshotPromptControlV3(_ComfyNodeBase):
    """Observe supplied conditionings and hooks without changing them."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare paired conditioning passthrough and JSON evidence."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.SnapshotPromptControl",
            display_name="Benchmark Snapshot Prompt Control",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.Conditioning.Input("positive"),
                _comfy_io.Conditioning.Input("negative"),
                _comfy_io.String.Input("case_id"),
                _comfy_io.String.Input("adapter_identities_json"),
                _comfy_io.Hooks.Input("hooks", optional=True),
            ],
            outputs=[
                _comfy_io.Conditioning.Output("positive"),
                _comfy_io.Conditioning.Output("negative"),
                _comfy_io.String.Output("snapshot_json"),
            ],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        positive: Sequence[object],
        negative: Sequence[object],
        case_id: str,
        adapter_identities_json: str,
        hooks: object | None = None,
    ) -> Any:
        """Return complete normalized metadata while preserving inputs."""

        identities = decode_adapter_identities(adapter_identities_json)
        snapshot = {
            "case_id": case_id,
            "positive": snapshot_conditioning(positive, identities),
            "negative": snapshot_conditioning(negative, identities),
            "hooks": snapshot_hook_group(hooks, identities),
        }
        encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        return _comfy_io.NodeOutput(
            positive,
            negative,
            encoded,
            ui={"prompt_control_snapshot": [snapshot]},
        )


def snapshot_conditioning(
    conditioning: Sequence[object], adapter_identities: tuple[str, ...]
) -> list[dict[str, object]]:
    """Normalize every conditioning tensor and metadata key in declared order."""

    result: list[dict[str, object]] = []
    for index, entry in enumerate(conditioning):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise TypeError(f"conditioning[{index}] must contain tensor and metadata.")
        tensor, metadata = entry
        if not isinstance(tensor, torch.Tensor) or not isinstance(metadata, dict):
            raise TypeError(f"conditioning[{index}] has invalid tensor or metadata.")
        if not all(isinstance(key, str) for key in metadata):
            raise TypeError(f"conditioning[{index}] metadata keys must be strings.")
        result.append(
            {
                "index": index,
                "tensor": snapshot_tensor(tensor),
                "metadata": {
                    key: _metadata_value(value, adapter_identities)
                    for key, value in sorted(metadata.items())
                },
            }
        )
    return result


def snapshot_hook_group(
    hook_group: object | None, adapter_identities: tuple[str, ...]
) -> list[dict[str, object]]:
    """Record ordered native hooks, base strengths, and every keyframe."""

    hooks = list(getattr(hook_group, "hooks", [])) if hook_group is not None else []
    if len(hooks) > len(adapter_identities):
        raise ValueError("Observed more hooks than declared adapter identities.")
    result: list[dict[str, object]] = []
    for index, hook in enumerate(hooks):
        keyframe_group = getattr(hook, "hook_keyframe", None)
        keyframes = list(getattr(keyframe_group, "keyframes", []))
        result.append(
            {
                "identity": adapter_identities[index],
                "order": index,
                "hook_type": type(hook).__name__,
                "hook_ref": _hook_ref(getattr(hook, "hook_ref", None), index),
                "hook_id": getattr(hook, "hook_id", None),
                "hook_scope": _enum_value(getattr(hook, "hook_scope", None)),
                "base_strength_model": _optional_number(
                    getattr(hook, "_strength_model", None)
                ),
                "base_strength_clip": _optional_number(
                    getattr(hook, "_strength_clip", None)
                ),
                "effective_strength_model": _optional_number(
                    getattr(hook, "strength_model", None)
                ),
                "effective_strength_clip": _optional_number(
                    getattr(hook, "strength_clip", None)
                ),
                "keyframes": [
                    {
                        "start_percent": float(keyframe.start_percent),
                        "strength": float(keyframe.strength),
                        "guarantee_steps": int(keyframe.guarantee_steps),
                    }
                    for keyframe in keyframes
                ],
            }
        )
    return result


def decode_adapter_identities(value: str) -> tuple[str, ...]:
    """Decode unique ordered adapter identities from JSON."""

    decoded: object = json.loads(value)
    if not isinstance(decoded, list) or not all(
        isinstance(item, str) and item for item in decoded
    ):
        raise ValueError("Adapter identities must be a JSON array of nonempty strings.")
    identities = tuple(decoded)
    if len(set(identities)) != len(identities):
        raise ValueError("Adapter identities must be unique.")
    return identities


def _metadata_value(value: object, identities: tuple[str, ...]) -> object:
    """Normalize the finite metadata surface emitted by Prompt Control and Comfy."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, torch.Tensor):
        return snapshot_tensor(value)
    if hasattr(value, "hooks"):
        return {"hook_group": snapshot_hook_group(value, identities)}
    enum_value = _enum_value(value)
    if enum_value is not None:
        return enum_value
    raise TypeError(f"Unsupported conditioning metadata value: {type(value).__name__}.")


def _hook_ref(value: object, index: int) -> str:
    """Preserve Prompt Control string refs and normalize native opaque refs."""

    return value if isinstance(value, str) else f"opaque-hook-ref-{index}"


def _enum_value(value: object) -> str | None:
    """Normalize enum-like values while rejecting arbitrary objects."""

    member = getattr(value, "value", None)
    return member if isinstance(member, str) else None


def _optional_number(value: object) -> float | None:
    """Narrow an optional scalar without accepting booleans."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Hook strength must be numeric or absent.")
    return float(value)
