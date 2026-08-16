# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Profile exactly one model call inside benchmark-only Comfy execution."""

from __future__ import annotations

import json
import threading
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile

from .indexed_call_capture import IndexedModelCallCapture
from .python_call_profile import PYTHON_CALL_PROFILER

_comfy_api: Any = None
if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for benchmark-only Comfy v3 nodes."""

        pass

else:
    _comfy_api = import_module("comfy_api.latest")
    _ComfyNodeBase = _comfy_api.io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else _comfy_api.io
_CAPTURES: dict[str, dict[str, object] | None] = {}
_CAPTURE_LOCK = threading.Lock()


class ProfileIndexedModelCallV3(_ComfyNodeBase):
    """Clone a model and profile one indexed underlying denoiser call."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare model, capture identity, and external trace path."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.ProfileIndexedModelCall",
            display_name="Benchmark Profile Indexed Model Call",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.Model.Input("model"),
                _comfy_io.String.Input("run_id"),
                _comfy_io.String.Input("trace_path"),
                _comfy_io.Int.Input("call_index", default=1, min=1, max=10000),
            ],
            outputs=[_comfy_io.Model.Output("model")],
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        run_id: str,
        trace_path: str,
        call_index: int,
    ) -> Any:
        """Install one indexed single-use profiler around the model wrapper."""

        trace = _validated_trace_path(trace_path)
        cloned = model.clone()
        existing_wrapper = cloned.model_options.get("model_function_wrapper")
        with _CAPTURE_LOCK:
            if run_id in _CAPTURES:
                raise ValueError(f"Operator profile is already active: {run_id!r}.")
            _CAPTURES[run_id] = None
        capture = IndexedModelCallCapture(call_index)

        def profile_model_call(apply_model: Any, args: dict[str, Any]) -> torch.Tensor:
            """Profile only the indexed call and delegate every other call."""

            if not capture.admit_next():
                return _delegate(existing_wrapper, apply_model, args)
            with torch_profile(
                activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
                profile_memory=True,
                record_shapes=True,
                with_stack=False,
            ) as profiler:
                python_profile = PYTHON_CALL_PROFILER.capture(
                    lambda: _delegate(existing_wrapper, apply_model, args)
                )
                result = python_profile.result
            profiler.export_chrome_trace(str(trace))
            rows = tuple(_operator_row(event) for event in profiler.key_averages())
            payload: dict[str, object] = {
                "run_id": run_id,
                "trace_path": str(trace),
                "call_index": call_index,
                "operators": sorted(
                    rows,
                    key=lambda row: (
                        -cast(float, row["self_device_time_us"]),
                        -cast(float, row["self_cpu_time_us"]),
                        str(row["name"]),
                    ),
                ),
                "python_functions": python_profile.functions,
            }
            with _CAPTURE_LOCK:
                _CAPTURES[run_id] = payload
            return result

        cloned.set_model_unet_function_wrapper(profile_model_call)
        return _comfy_io.NodeOutput(cloned)


class ReadOperatorProfileV3(_ComfyNodeBase):
    """Expose and release one completed indexed-call operator profile."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the sampled latent dependency and JSON result."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.ReadOperatorProfile",
            display_name="Benchmark Read Operator Profile",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.Latent.Input("latent"),
                _comfy_io.String.Input("run_id"),
            ],
            outputs=[
                _comfy_io.Latent.Output("latent"),
                _comfy_io.String.Output("profile_json"),
            ],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, latent: dict[str, Any], run_id: str) -> Any:
        """Return one complete capture and remove its task-local state."""

        with _CAPTURE_LOCK:
            capture = _CAPTURES.pop(run_id, None)
        if capture is None:
            raise ValueError(f"Operator profile did not complete: {run_id!r}.")
        encoded = json.dumps(capture, sort_keys=True, separators=(",", ":"))
        return _comfy_io.NodeOutput(
            latent,
            encoded,
            ui={"operator_profile": [capture]},
        )


def _delegate(
    existing_wrapper: Any,
    apply_model: Any,
    args: dict[str, Any],
) -> torch.Tensor:
    """Call the preserved wrapper or native model function exactly once."""

    if existing_wrapper is not None:
        value = existing_wrapper(apply_model, args)
    else:
        conditioning = args.get("c", {})
        if not isinstance(conditioning, dict):
            raise TypeError("Profiled model conditioning must be a dictionary.")
        value = apply_model(args["input"], args["timestep"], **conditioning)
    if not isinstance(value, torch.Tensor):
        raise TypeError("Profiled model call must return a tensor.")
    return value


def _validated_trace_path(value: str) -> Path:
    """Require an absolute JSON trace under an existing directory."""

    path = Path(value)
    if not path.is_absolute() or path.suffix.lower() != ".json":
        raise ValueError("Operator trace path must be an absolute JSON path.")
    resolved_parent = path.parent.resolve()
    if not resolved_parent.is_dir():
        raise ValueError("Operator trace parent directory must exist.")
    return resolved_parent / path.name


def _operator_row(event: Any) -> dict[str, object]:
    """Decode one complete stable PyTorch aggregate row."""

    return {
        "name": str(event.key),
        "count": int(event.count),
        "self_cpu_time_us": float(event.self_cpu_time_total),
        "total_cpu_time_us": float(event.cpu_time_total),
        "self_device_time_us": float(event.self_device_time_total),
        "total_device_time_us": float(event.device_time_total),
        "self_cpu_memory_bytes": int(event.self_cpu_memory_usage),
        "total_cpu_memory_bytes": int(event.cpu_memory_usage),
        "self_device_memory_bytes": int(event.self_device_memory_usage),
        "total_device_memory_bytes": int(event.device_memory_usage),
    }
