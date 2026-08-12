"""Observe Prompt Control UUIDs and WeightHook transitions during sampling."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any

import torch

from .prompt_control_snapshot import (
    decode_adapter_identities,
    snapshot_hook_group,
)

_comfy_api: Any = None
if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for benchmark-only runtime nodes."""

        pass

else:
    _comfy_api = import_module("comfy_api.latest")
    _ComfyNodeBase = _comfy_api.io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else _comfy_api.io


@dataclass
class _PromptControlState:
    """Track one sampling execution's runtime-only Prompt Control evidence."""

    identities: tuple[str, ...]
    static_patch_target_count: int
    static_patch_entry_count_histogram: dict[str, int]
    model_call_count: int = 0
    registered_hooks: list[dict[str, object]] = field(default_factory=list)
    schedule_transitions: list[dict[str, object]] = field(default_factory=list)
    calls: list[dict[str, object]] = field(default_factory=list)
    hook_index_by_ref: dict[int, int] = field(default_factory=dict)
    pending_strengths: tuple[float, ...] = ()
    last_strengths: tuple[float, ...] | None = None


_STATES: dict[str, _PromptControlState] = {}
_LOCK = threading.Lock()


class InstrumentPromptControlModelV3(_ComfyNodeBase):
    """Instrument one model clone without changing Prompt Control behavior."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the model, run identity, and adapter labels."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.InstrumentPromptControlModel",
            display_name="Benchmark Instrument Prompt Control Model",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.Model.Input("model"),
                _comfy_io.String.Input("run_id"),
                _comfy_io.String.Input("adapter_identities_json"),
            ],
            outputs=[_comfy_io.Model.Output("model")],
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, model: Any, run_id: str, adapter_identities_json: str) -> Any:
        """Install keyed hook callbacks and a preserving model wrapper."""

        identities = decode_adapter_identities(adapter_identities_json)
        cloned = model.clone()
        existing_wrapper = cloned.model_options.get("model_function_wrapper")
        patch_lists = list(getattr(cloned, "patches", {}).values())
        entry_histogram: dict[str, int] = {}
        for entries in patch_lists:
            if not isinstance(entries, list):
                raise TypeError("Prompt Control static model patches must be lists.")
            key = str(len(entries))
            entry_histogram[key] = entry_histogram.get(key, 0) + 1
        state = _PromptControlState(
            identities,
            static_patch_target_count=len(patch_lists),
            static_patch_entry_count_histogram=entry_histogram,
        )
        with _LOCK:
            if run_id in _STATES:
                raise ValueError(f"Prompt Control probe run is active: {run_id!r}.")
            _STATES[run_id] = state

        def observe_registered_hooks(runtime_model: Any, hooks: Any, *_: Any) -> None:
            del runtime_model
            raw_hooks = list(getattr(hooks, "hooks", []))
            if len(raw_hooks) != len(identities):
                raise ValueError(
                    "Prompt Control registered hook count must match "
                    "declared identities."
                )
            state.hook_index_by_ref = {
                id(getattr(hook, "hook_ref", None)): index
                for index, hook in enumerate(raw_hooks)
            }
            snapshots = snapshot_hook_group(hooks, identities)
            state.registered_hooks = snapshots
            state.pending_strengths = _strength_vector(hooks, state)

        def observe_applied_hooks(runtime_model: Any, hooks: Any) -> None:
            del runtime_model
            state.pending_strengths = _strength_vector(hooks, state)

        add_callback = getattr(cloned, "add_callback_with_key", None)
        if not callable(add_callback):
            raise TypeError("Prompt Control probe requires keyed patcher callbacks.")
        add_callback(
            "on_register_all_hook_patches",
            f"simple_syrup_pc_register_{run_id}",
            observe_registered_hooks,
        )
        add_callback(
            "on_apply_hooks",
            f"simple_syrup_pc_apply_{run_id}",
            observe_applied_hooks,
        )

        def observe_model_call(apply_model: Any, args: dict[str, Any]) -> torch.Tensor:
            """Delegate one call and record sampler-assigned UUIDs and hooks."""

            if existing_wrapper is not None:
                output = existing_wrapper(apply_model, args)
            else:
                conditioning = args.get("c", {})
                if not isinstance(conditioning, dict):
                    raise TypeError(
                        "Prompt Control runtime conditioning must be a dict."
                    )
                output = apply_model(args["input"], args["timestep"], **conditioning)
            if not isinstance(output, torch.Tensor):
                raise TypeError("Prompt Control model wrapper must return a tensor.")
            state.model_call_count += 1
            transformer_options = _transformer_options(args)
            uuids = transformer_options.get("uuids", [])
            if not isinstance(uuids, list):
                raise TypeError("Comfy transformer UUIDs must be a list.")
            cond_or_uncond = transformer_options.get("cond_or_uncond", [])
            if not isinstance(cond_or_uncond, list):
                raise TypeError("Comfy cond_or_uncond must be a list.")
            state.calls.append(
                {
                    "call_index": state.model_call_count,
                    "timestep": _first_scalar(args.get("timestep")),
                    "uuids": [str(value) for value in uuids],
                    "cond_or_uncond": list(cond_or_uncond),
                    "effective_strengths": list(state.pending_strengths),
                }
            )
            if state.pending_strengths != state.last_strengths:
                state.schedule_transitions.append(state.calls[-1].copy())
                state.last_strengths = state.pending_strengths
            return output

        cloned.set_model_unet_function_wrapper(observe_model_call)
        return _comfy_io.NodeOutput(cloned)


class ReadPromptControlRuntimeV3(_ComfyNodeBase):
    """Finalize runtime UUID and hook transition evidence."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare latent passthrough and JSON output."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.ReadPromptControlRuntime",
            display_name="Benchmark Read Prompt Control Runtime",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.Latent.Input("latent"),
                _comfy_io.String.Input("run_id"),
            ],
            outputs=[
                _comfy_io.Latent.Output("latent"),
                _comfy_io.String.Output("runtime_json"),
            ],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, latent: dict[str, Any], run_id: str) -> Any:
        """Return and release one completed runtime observation."""

        with _LOCK:
            state = _STATES.pop(run_id, None)
        if state is None:
            raise ValueError(f"Prompt Control run was not instrumented: {run_id!r}.")
        metrics = {
            "run_id": run_id,
            "model_call_count": state.model_call_count,
            "static_patch_target_count": state.static_patch_target_count,
            "static_patch_entry_count_histogram": (
                state.static_patch_entry_count_histogram
            ),
            "registered_hooks": state.registered_hooks,
            "schedule_transitions": state.schedule_transitions,
            "calls": state.calls,
        }
        encoded = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
        return _comfy_io.NodeOutput(
            latent,
            encoded,
            ui={"prompt_control_runtime": [metrics]},
        )


def _transformer_options(args: dict[str, Any]) -> dict[str, Any]:
    """Narrow the model wrapper's transformer options."""

    conditioning = args.get("c", {})
    if not isinstance(conditioning, dict):
        raise TypeError("Prompt Control runtime conditioning must be a dict.")
    options = conditioning.get("transformer_options", {})
    if not isinstance(options, dict):
        raise TypeError("Prompt Control transformer_options must be a dict.")
    return options


def _first_scalar(value: object) -> float:
    """Read one timestep scalar from the dynamic wrapper arguments."""

    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        raise TypeError("Prompt Control timestep must be a nonempty tensor.")
    return float(value.detach().flatten()[0].cpu().item())


def _required_number(value: object) -> float:
    """Narrow one required hook strength."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Prompt Control effective hook strength must be numeric.")
    return float(value)


def _strength_vector(
    hook_group: object, state: _PromptControlState
) -> tuple[float, ...]:
    """Map one active hook subset to the declared adapter order."""

    strengths = [0.0] * len(state.identities)
    for hook in list(getattr(hook_group, "hooks", [])):
        ref = id(getattr(hook, "hook_ref", None))
        index = state.hook_index_by_ref.get(ref)
        if index is None:
            raise ValueError("Prompt Control applied an unregistered WeightHook.")
        strengths[index] = _required_number(getattr(hook, "strength_model", None))
    return tuple(strengths)
