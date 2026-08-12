"""Publish read-only benchmark evidence for installed MODEL modifiers."""

from __future__ import annotations

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


class SnapshotModelModifierV3(_ComfyNodeBase):
    """Pass MODEL through while publishing stable modifier-state evidence."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare one dev-only pass-through MODEL evidence node."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.SnapshotModelModifier",
            display_name="Benchmark Snapshot Model Modifier",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.Model.Input("model"),
                _comfy_io.String.Input("run_id"),
            ],
            outputs=[_comfy_io.Model.Output("model")],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, model: object, run_id: str) -> Any:
        """Return the exact MODEL and one immutable UI snapshot."""

        snapshot = snapshot_model_modifier_state(model, run_id=run_id)
        return _comfy_io.NodeOutput(
            model,
            ui={"model_modifier_snapshot": [snapshot]},
        )


def snapshot_model_modifier_state(
    model: object,
    *,
    run_id: str,
) -> dict[str, object]:
    """Describe stable upstream modifier surfaces without changing MODEL state."""

    if not isinstance(run_id, str) or not run_id:
        raise ValueError("MODEL modifier snapshot run ID must not be empty.")
    model_options = _dictionary_attribute(model, "model_options")
    transformer_options = _nested_dictionary(
        model_options,
        "transformer_options",
        owner="MODEL",
    )
    wrappers = _dictionary_attribute(model, "wrappers")
    object_patches = _dictionary_attribute(model, "object_patches")
    model_wrapper = _optional_callable_name(
        model_options.get("model_function_wrapper"),
        field="model_function_wrapper",
    )
    optimized_attention = _optional_callable_name(
        transformer_options.get("optimized_attention_override"),
        field="optimized_attention_override",
    )
    marker = model_options.get("ppm_negpip", False)
    if not isinstance(marker, bool):
        raise TypeError("MODEL ppm_negpip marker must be boolean.")
    cache_holder = transformer_options.get("easycache")
    patches = transformer_options.get("patches", {})
    if not isinstance(patches, dict):
        raise TypeError("MODEL transformer patches must be a dictionary.")
    return {
        "run_id": run_id,
        "cache_holder_type": (
            None if cache_holder is None else type(cache_holder).__qualname__
        ),
        "model_function_wrapper": model_wrapper,
        "optimized_attention_override": optimized_attention,
        "ppm_negpip": marker,
        "wrappers": _wrapper_records(wrappers),
        "object_patch_keys": _sorted_string_keys(
            object_patches,
            owner="MODEL object patches",
        ),
        "transformer_patch_counts": _patch_counts(patches),
    }


def _wrapper_records(mapping: dict[object, object]) -> list[dict[str, object]]:
    """Return wrapper types and keys in exact installed insertion order."""

    records: list[dict[str, object]] = []
    for wrapper_type, keyed in mapping.items():
        if not isinstance(wrapper_type, str) or not isinstance(keyed, dict):
            raise TypeError("MODEL wrappers must map string types to dictionaries.")
        for key, callbacks in keyed.items():
            if not isinstance(callbacks, list) or any(
                not callable(callback) for callback in callbacks
            ):
                raise TypeError("MODEL wrapper callbacks must be callable lists.")
            records.append(
                {
                    "wrapper_type": wrapper_type,
                    "key": _stable_key(key),
                    "callbacks": [_callable_name(callback) for callback in callbacks],
                }
            )
    return records


def _patch_counts(mapping: dict[object, object]) -> dict[str, int]:
    """Return sorted attention-patch cardinalities without callback identities."""

    counts: dict[str, int] = {}
    for key, callbacks in mapping.items():
        if not isinstance(key, str) or not isinstance(callbacks, list):
            raise TypeError("MODEL transformer patches must map strings to lists.")
        counts[key] = len(callbacks)
    return dict(sorted(counts.items()))


def _sorted_string_keys(mapping: dict[object, object], *, owner: str) -> list[str]:
    """Return stable sorted keys after rejecting non-string host state."""

    if any(not isinstance(key, str) for key in mapping):
        raise TypeError(f"{owner} keys must be strings.")
    return sorted(str(key) for key in mapping)


def _stable_key(value: object) -> str:
    """Return one wrapper key without unstable object representations."""

    if isinstance(value, str):
        return value
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    return type(value).__qualname__


def _optional_callable_name(value: object, *, field: str) -> str | None:
    """Return one optional callable name after fail-closed validation."""

    if value is None:
        return None
    if not callable(value):
        raise TypeError(f"MODEL {field} must be callable.")
    return _callable_name(value)


def _callable_name(value: object) -> str:
    """Return the most specific stable callable name available."""

    name = getattr(value, "__qualname__", None)
    return name if isinstance(name, str) and name else type(value).__qualname__


def _dictionary_attribute(owner: object, name: str) -> dict[object, object]:
    """Return one required host dictionary without coercion."""

    value = getattr(owner, name, None)
    if not isinstance(value, dict):
        raise TypeError(f"MODEL {name} must be a dictionary.")
    return value


def _nested_dictionary(
    mapping: dict[object, object],
    key: str,
    *,
    owner: str,
) -> dict[object, object]:
    """Return one required nested dictionary without changing it."""

    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{owner} {key} must be a dictionary.")
    return value
