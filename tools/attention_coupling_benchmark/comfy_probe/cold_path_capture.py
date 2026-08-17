# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Capture standard-UNet cold-stage diagnostics for one benchmark request."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

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
_LOGGER_NAME = "simple_syrup.runtime.regional_lora.standard_unet_cold_path"
_COMPOSITION_LOGGER_NAME = (
    "simple_syrup.runtime.regional_lora.standard_unet_composition"
)


class _ColdPathHandler(logging.Handler):
    """Retain every ordered JSON-safe cold-stage diagnostic."""

    def __init__(self) -> None:
        """Initialize empty thread-safe record storage."""

        super().__init__(logging.DEBUG)
        self._lock = threading.Lock()
        self._records: list[str] = []
        self._model_call_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Capture only the focused structured payload."""

        if record.name == _COMPOSITION_LOGGER_NAME:
            if getattr(record, "regional_composition", None) is not None:
                with self._lock:
                    self._model_call_count += 1
            return
        if record.name != _LOGGER_NAME:
            return
        payload = getattr(record, "cold_path_diagnostics", None)
        if payload is None:
            return
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if not isinstance(json.loads(serialized), dict):
            raise TypeError("Cold-path diagnostic payload must be an object.")
        with self._lock:
            self._records.append(serialized)

    def records(self) -> list[dict[str, object]]:
        """Return detached ordered JSON objects."""

        with self._lock:
            return [
                cast(dict[str, object], json.loads(serialized))
                for serialized in self._records
            ]

    @property
    def model_call_count(self) -> int:
        """Return the exact observed persistent-composition call count."""

        with self._lock:
            return self._model_call_count


@dataclass(frozen=True, slots=True)
class _CaptureState:
    """Retain the sole active logger lease and handler."""

    loggers: tuple[tuple[logging.Logger, int], ...]
    handler: _ColdPathHandler


_STATES: dict[str, _CaptureState] = {}
_STATE_LOCK = threading.Lock()


class CaptureColdPathDiagnosticsV3(_ComfyNodeBase):
    """Start one exclusive cold-stage capture before sampler preparation."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare a benchmark-only transparent MODEL boundary."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.CaptureColdPathDiagnostics",
            display_name="Benchmark Capture Cold Path Diagnostics",
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
        """Lease DEBUG capture without cloning or mutating the supplied MODEL."""

        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Cold-path capture run id must be non-empty.")
        loggers = tuple(
            logging.getLogger(name) for name in (_LOGGER_NAME, _COMPOSITION_LOGGER_NAME)
        )
        handler = _ColdPathHandler()
        with _STATE_LOCK:
            if _STATES:
                raise RuntimeError("Cold-path diagnostic capture is already active.")
            state = _CaptureState(
                tuple((logger, logger.level) for logger in loggers),
                handler,
            )
            _STATES[run_id] = state
            for logger in loggers:
                logger.setLevel(logging.DEBUG)
                logger.addHandler(handler)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        return _comfy_io.NodeOutput(model)


class ReadColdPathDiagnosticsV3(_ComfyNodeBase):
    """Synchronize and publish one complete cold-stage capture."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the benchmark-only latent and JSON terminal."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.ReadColdPathDiagnostics",
            display_name="Benchmark Read Cold Path Diagnostics",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.Latent.Input("latent"),
                _comfy_io.String.Input("run_id"),
            ],
            outputs=[
                _comfy_io.Latent.Output("latent"),
                _comfy_io.String.Output("diagnostics_json"),
            ],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, latent: dict[str, Any], run_id: str) -> Any:
        """End the logger lease after all queued CUDA work completes."""

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        completed_at_ns = time.perf_counter_ns()
        with _STATE_LOCK:
            state = _STATES.pop(run_id, None)
            if state is None:
                raise ValueError(
                    f"Cold-path diagnostic capture was not started: {run_id!r}."
                )
            for logger, original_level in state.loggers:
                logger.removeHandler(state.handler)
                logger.setLevel(original_level)
        records = state.handler.records()
        if not records:
            raise ValueError("Cold-path diagnostic capture observed no stages.")
        result: dict[str, object] = {
            "run_id": run_id,
            "completed_at_ns": completed_at_ns,
            "record_count": len(records),
            "records": records,
            "model_call_count": state.handler.model_call_count,
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            ),
        }
        encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
        return _comfy_io.NodeOutput(
            latent,
            encoded,
            ui={"cold_path_diagnostics": [result]},
        )
