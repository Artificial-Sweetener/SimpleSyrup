"""Measure model-call count, sampling runtime, and peak CUDA memory."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any

import torch

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
class _ProbeState:
    """Track actual wrapped model calls and sampling-resource measurements."""

    started_at: float
    model_call_count: int = 0
    model_input_batch_sizes: list[int] = field(default_factory=list)


_STATES: dict[str, _ProbeState] = {}
_STATE_LOCK = threading.Lock()


class InstrumentModelV3(_ComfyNodeBase):
    """Clone a model and count every call reaching its preserved base wrapper."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the benchmark model instrumentation inputs and output."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.InstrumentModel",
            display_name="Benchmark Instrument Model",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.Model.Input("model"),
                _comfy_io.String.Input("run_id"),
            ],
            outputs=[_comfy_io.Model.Output("model")],
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, model: Any, run_id: str) -> Any:
        """Install a counting wrapper while preserving any existing wrapper."""

        cloned = model.clone()
        existing_wrapper = cloned.model_options.get("model_function_wrapper")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        state = _ProbeState(started_at=time.perf_counter())
        with _STATE_LOCK:
            if run_id in _STATES:
                raise ValueError(f"Benchmark run is already active: {run_id!r}.")
            _STATES[run_id] = state

        def count_model_call(apply_model: Any, args: dict[str, Any]) -> torch.Tensor:
            """Count and delegate one underlying model-function invocation."""

            model_input = args.get("input")
            if not isinstance(model_input, torch.Tensor):
                raise TypeError("Benchmark model input must be a tensor.")
            if model_input.ndim < 1 or int(model_input.shape[0]) < 1:
                raise ValueError("Benchmark model input batch must be positive.")
            batch_size = int(model_input.shape[0])
            with _STATE_LOCK:
                state.model_call_count += 1
                state.model_input_batch_sizes.append(batch_size)
            if existing_wrapper is not None:
                return _tensor_result(existing_wrapper(apply_model, args))
            conditioning = args.get("c", {})
            if not isinstance(conditioning, dict):
                raise TypeError("Benchmark model conditioning must be a dictionary.")
            return _tensor_result(
                apply_model(args["input"], args["timestep"], **conditioning)
            )

        cloned.set_model_unet_function_wrapper(count_model_call)
        return _comfy_io.NodeOutput(cloned)


class ReadMetricsV3(_ComfyNodeBase):
    """Read and expose one completed benchmark run's exact probe metrics."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the measured latent passthrough and JSON output."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.ReadMetrics",
            display_name="Benchmark Read Metrics",
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
        """Finalize elapsed time, call count, and peak CUDA memory."""

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        completed_at = time.perf_counter()
        with _STATE_LOCK:
            state = _STATES.pop(run_id, None)
        if state is None:
            raise ValueError(f"Benchmark run was not instrumented: {run_id!r}.")
        peak_vram_bytes = (
            int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        )
        metrics: dict[str, object] = {
            "run_id": run_id,
            "runtime_ms": (completed_at - state.started_at) * 1000.0,
            "model_call_count": state.model_call_count,
            "model_input_batch_sizes": state.model_input_batch_sizes,
            "model_input_batch_elements": sum(state.model_input_batch_sizes),
            "peak_vram_bytes": peak_vram_bytes,
        }
        metrics_json = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
        return _comfy_io.NodeOutput(
            latent,
            metrics_json,
            ui={"benchmark_metrics": [metrics]},
        )


def _tensor_result(value: object) -> torch.Tensor:
    """Narrow the dynamic Comfy model-function result."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("Benchmark model wrapper must return a tensor.")
    return value
