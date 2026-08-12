"""Expose benchmark-only LoRA execution evidence through focused v3 nodes."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any

import torch

from .lora_patch_snapshot import (
    schedule_signature,
    snapshot_hook_patches,
    snapshot_static_patches,
)

_comfy_api: Any = None
if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for benchmark-only Comfy v3 nodes."""

        pass

else:
    _comfy_api = import_module("comfy_api.latest")
    _ComfyNodeBase = _comfy_api.io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else _comfy_api.io


@dataclass
class _LoraProbeState:
    """Track one LoRA sampling execution without owning its workflow."""

    started_at: float
    model: Any
    adapter_identities: tuple[str, ...]
    capture_outputs: bool
    static_patches: dict[str, object]
    model_call_count: int = 0
    hook_patches: list[dict[str, object]] = field(default_factory=list)
    schedule_observations: list[dict[str, object]] = field(default_factory=list)
    denoiser_outputs: list[dict[str, object]] = field(default_factory=list)
    pending_schedule_signature: tuple[float, ...] | None = None
    last_schedule_signature: tuple[float, ...] | None = None


_STATES: dict[str, _LoraProbeState] = {}
_STATE_LOCK = threading.Lock()


class InstrumentLoraModelV3(_ComfyNodeBase):
    """Instrument a model without changing static or scheduled LoRA math."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare LoRA instrumentation inputs and the preserved model output."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.InstrumentLoraModel",
            display_name="Benchmark Instrument LoRA Model",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.Model.Input("model"),
                _comfy_io.String.Input("run_id"),
                _comfy_io.String.Input("adapter_identities_json"),
                _comfy_io.Boolean.Input("capture_outputs", default=False),
            ],
            outputs=[_comfy_io.Model.Output("model")],
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        run_id: str,
        adapter_identities_json: str,
        capture_outputs: bool,
    ) -> Any:
        """Install observation around the model's preserved wrapper chain."""

        identities = _adapter_identities(adapter_identities_json)
        cloned = model.clone()
        existing_wrapper = cloned.model_options.get("model_function_wrapper")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        state = _LoraProbeState(
            started_at=time.perf_counter(),
            model=cloned,
            adapter_identities=identities,
            capture_outputs=capture_outputs,
            static_patches=snapshot_static_patches(cloned, identities),
        )
        with _STATE_LOCK:
            if run_id in _STATES:
                raise ValueError(f"LoRA benchmark run is already active: {run_id!r}.")
            _STATES[run_id] = state

        def observe_registered_hooks(
            runtime_model: Any,
            hooks: Any,
            target_dict: Any,
            model_options: Any,
            registered: Any,
        ) -> None:
            """Capture patches after Comfy registers them on its runtime clone."""

            del target_dict, model_options, registered
            state.hook_patches = snapshot_hook_patches(runtime_model, identities, hooks)
            state.pending_schedule_signature = (
                schedule_signature(state.hook_patches) if state.hook_patches else None
            )

        def observe_applied_hooks(runtime_model: Any, hooks: Any) -> None:
            """Capture each effective WeightHook schedule transition."""

            state.hook_patches = snapshot_hook_patches(runtime_model, identities, hooks)
            state.pending_schedule_signature = (
                schedule_signature(state.hook_patches) if state.hook_patches else None
            )

        add_callback = getattr(cloned, "add_callback_with_key", None)
        if not callable(add_callback):
            raise TypeError("LoRA probe model must support keyed patcher callbacks.")
        add_callback(
            "on_register_all_hook_patches",
            f"simple_syrup_benchmark_lora_register_{run_id}",
            observe_registered_hooks,
        )
        add_callback(
            "on_apply_hooks",
            f"simple_syrup_benchmark_lora_apply_{run_id}",
            observe_applied_hooks,
        )

        def observe_model_call(apply_model: Any, args: dict[str, Any]) -> torch.Tensor:
            """Delegate one model call and record only requested evidence."""

            if existing_wrapper is not None:
                output = _tensor_result(existing_wrapper(apply_model, args))
            else:
                conditioning = args.get("c", {})
                if not isinstance(conditioning, dict):
                    raise TypeError("LoRA probe conditioning must be a dictionary.")
                output = _tensor_result(
                    apply_model(args["input"], args["timestep"], **conditioning)
                )
            state.model_call_count += 1
            signature = state.pending_schedule_signature
            if signature is not None and signature != state.last_schedule_signature:
                state.schedule_observations.append(
                    {
                        "call_index": state.model_call_count,
                        "timestep": _first_scalar(args.get("timestep")),
                        "effective_strengths": list(signature),
                    }
                )
                state.last_schedule_signature = signature
            if capture_outputs:
                state.denoiser_outputs.append(
                    _output_digest(output, state.model_call_count)
                )
            return output

        cloned.set_model_unet_function_wrapper(observe_model_call)
        return _comfy_io.NodeOutput(cloned)


class ReadLoraMetricsV3(_ComfyNodeBase):
    """Finalize one LoRA execution's patch and denoiser evidence."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare latent passthrough and JSON evidence outputs."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.ReadLoraMetrics",
            display_name="Benchmark Read LoRA Metrics",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.Latent.Input("latent"),
                _comfy_io.String.Input("run_id"),
            ],
            outputs=[
                _comfy_io.Latent.Output("latent"),
                _comfy_io.String.Output("metrics_json"),
            ],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, latent: dict[str, Any], run_id: str) -> Any:
        """Return exact evidence and release the run-scoped probe state."""

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        completed_at = time.perf_counter()
        with _STATE_LOCK:
            state = _STATES.pop(run_id, None)
        if state is None:
            raise ValueError(f"LoRA benchmark run was not instrumented: {run_id!r}.")
        metrics: dict[str, object] = {
            "run_id": run_id,
            "capture_outputs": state.capture_outputs,
            "runtime_ms": (completed_at - state.started_at) * 1000.0,
            "model_call_count": state.model_call_count,
            "peak_vram_bytes": (
                int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else 0
            ),
            "static_patches": state.static_patches,
            "hook_patches": state.hook_patches,
            "schedule_observations": state.schedule_observations,
            "denoiser_outputs": state.denoiser_outputs,
        }
        metrics_json = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
        return _comfy_io.NodeOutput(
            latent, metrics_json, ui={"lora_benchmark_metrics": [metrics]}
        )


def _adapter_identities(value: str) -> tuple[str, ...]:
    """Decode unique ordered adapter identities from JSON."""

    decoded: object = json.loads(value)
    if not isinstance(decoded, list) or not all(
        isinstance(item, str) and item for item in decoded
    ):
        raise ValueError("Adapter identities must be a JSON array of nonempty strings.")
    identities = tuple(decoded)
    if len(set(identities)) != len(identities):
        raise ValueError("Adapter identities must be unique within a run.")
    return identities


def _output_digest(output: torch.Tensor, call_index: int) -> dict[str, object]:
    """Hash one denoiser output in canonical float32 CPU representation."""

    canonical = output.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return {
        "call_index": call_index,
        "shape": list(output.shape),
        "source_dtype": str(output.dtype),
        "float32_sha256": hashlib.sha256(canonical.numpy().tobytes()).hexdigest(),
    }


def _first_scalar(value: object) -> float:
    """Read the first timestep scalar from Comfy's dynamic call arguments."""

    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        raise TypeError("LoRA probe timestep must be a nonempty tensor.")
    return float(value.detach().flatten()[0].cpu().item())


def _tensor_result(value: object) -> torch.Tensor:
    """Narrow the dynamic Comfy model-function result."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("LoRA model wrapper must return a tensor.")
    return value
