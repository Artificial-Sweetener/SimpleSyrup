"""Normalize dynamic Comfy LoRA patch state into stable JSON evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def snapshot_static_patches(
    model: Any, adapter_identities: Sequence[str]
) -> dict[str, object]:
    """Record static patch targets and prove consistent adapter ordering."""

    patches = _mapping(getattr(model, "patches", {}), "model.patches")
    target_keys = sorted(_string_keys(patches, "model.patches"))
    signatures = {
        key: _ordered_patch_signatures(
            patches[key], adapter_identities, f"model.patches[{key!r}]"
        )
        for key in target_keys
    }
    canonical_target = target_keys[0] if target_keys else None
    canonical_order = (
        signatures[canonical_target] if canonical_target is not None else []
    )
    inconsistent = [key for key in target_keys if signatures[key] != canonical_order]
    entry_counts = Counter(len(signatures[key]) for key in target_keys)
    return {
        "target_count": len(target_keys),
        "target_keys": target_keys,
        "entry_count_histogram": {
            str(count): target_count
            for count, target_count in sorted(entry_counts.items())
        },
        "canonical_target": canonical_target,
        "canonical_patch_order": canonical_order,
        "inconsistent_order_targets": inconsistent,
    }


def snapshot_hook_patches(
    model: Any,
    adapter_identities: Sequence[str],
    hook_group: object | None = None,
) -> list[dict[str, object]]:
    """Record registered hooks in their effective composition order."""

    selected_hooks = (
        hook_group if hook_group is not None else getattr(model, "current_hooks", None)
    )
    hooks = list(getattr(selected_hooks, "hooks", [])) if selected_hooks else []
    hook_patches = _mapping(getattr(model, "hook_patches", {}), "hook_patches")
    if len(hooks) > len(adapter_identities):
        raise ValueError("Observed more WeightHooks than declared adapter identities.")
    snapshots: list[dict[str, object]] = []
    for index, hook in enumerate(hooks):
        hook_ref = getattr(hook, "hook_ref", None)
        patches = _mapping(hook_patches.get(hook_ref, {}), "hook patches")
        target_keys = sorted(_string_keys(patches, "hook patches"))
        snapshots.append(
            {
                "identity": adapter_identities[index],
                "order": index,
                "base_strength_model": _number(
                    getattr(hook, "_strength_model", None), "base hook strength"
                ),
                "schedule_strength": _number(
                    getattr(hook, "strength", None), "hook schedule strength"
                ),
                "effective_strength_model": _number(
                    getattr(hook, "strength_model", None),
                    "effective hook strength",
                ),
                "target_count": len(target_keys),
                "target_keys": target_keys,
            }
        )
    return snapshots


def schedule_signature(hooks: Sequence[Mapping[str, object]]) -> tuple[float, ...]:
    """Return the ordered effective-strength signature for change detection."""

    signature: list[float] = []
    for hook in hooks:
        value = hook.get("effective_strength_model")
        if not isinstance(value, float):
            raise TypeError("Hook snapshot effective strength must be a float.")
        signature.append(value)
    return tuple(signature)


def _ordered_patch_signatures(
    value: object,
    adapter_identities: Sequence[str],
    field: str,
) -> list[dict[str, object]]:
    """Normalize one target's ordered static patch list."""

    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list.")
    if len(value) > len(adapter_identities):
        raise ValueError(f"{field} has more patches than declared identities.")
    result: list[dict[str, object]] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, tuple) or len(entry) < 3:
            raise TypeError(f"{field}[{index}] must be a Comfy patch tuple.")
        result.append(
            {
                "identity": adapter_identities[index],
                "order": index,
                "strength_patch": _number(entry[0], "patch strength"),
                "patch_type": type(entry[1]).__name__,
                "strength_model": _number(entry[2], "model strength"),
            }
        )
    return result


def _mapping(value: object, field: str) -> Mapping[object, object]:
    """Narrow one dynamic mapping boundary."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping.")
    return value


def _string_keys(value: Mapping[object, object], field: str) -> list[str]:
    """Validate and return string mapping keys."""

    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} keys must be strings.")
    return [key for key in value if isinstance(key, str)]


def _number(value: object, field: str) -> float:
    """Narrow one scalar number without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric.")
    return float(value)
