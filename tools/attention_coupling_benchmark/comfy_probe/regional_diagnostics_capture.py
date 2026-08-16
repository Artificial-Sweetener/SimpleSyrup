# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Capture structured regional diagnostics for managed benchmark workflows."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

_comfy_api: Any = None
if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for benchmark-only Comfy v3 nodes."""

        pass

else:
    _comfy_api = import_module("comfy_api.latest")
    _ComfyNodeBase = _comfy_api.io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else _comfy_api.io

_LOGGER_NAMES = frozenset(
    {
        "simple_syrup.runtime.attention_coupling.anima_diagnostics",
        "simple_syrup.runtime.attention_coupling.unet_attention_phase",
        "simple_syrup.runtime.regional_lora.standard_unet_composition",
        "simple_syrup.runtime.regional_lora.anima_diagnostics",
        "simple_syrup.runtime.attention_coupling.unet_diagnostics",
    }
)


class _RegionalDiagnosticsHandler(logging.Handler):
    """Retain JSON-safe unique snapshots and the exact emitted record count."""

    def __init__(self) -> None:
        """Initialize empty thread-safe capture state."""

        super().__init__(level=logging.DEBUG)
        self._lock = threading.Lock()
        self._record_count = 0
        self._serialized_snapshots: list[str] = []
        self._attention_phase_record_count = 0
        self._serialized_attention_phases: list[str] = []
        self._self_attention_record_count = 0
        self._serialized_self_attention: list[str] = []
        self._composition_record_count = 0
        self._serialized_composition: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Capture one target logger record without retaining sensitive objects."""

        if record.name not in _LOGGER_NAMES:
            return
        snapshot = getattr(record, "regional_diagnostics", None)
        attention_phase = getattr(record, "attention_phase_diagnostics", None)
        self_attention = getattr(record, "self_attention_diagnostics", None)
        composition = getattr(record, "regional_composition", None)
        if (
            snapshot is None
            and attention_phase is None
            and self_attention is None
            and composition is None
        ):
            return
        with self._lock:
            if snapshot is not None:
                serialized = _serialize_payload(snapshot, "Regional diagnostics")
                self._record_count += 1
                if not self._serialized_snapshots or (
                    self._serialized_snapshots[-1] != serialized
                ):
                    self._serialized_snapshots.append(serialized)
            if attention_phase is not None:
                serialized = _serialize_payload(
                    attention_phase,
                    "Attention phase diagnostics",
                )
                self._attention_phase_record_count += 1
                if not self._serialized_attention_phases or (
                    self._serialized_attention_phases[-1] != serialized
                ):
                    self._serialized_attention_phases.append(serialized)
            if self_attention is not None:
                serialized = _serialize_payload(
                    self_attention,
                    "Self-attention diagnostics",
                )
                self._self_attention_record_count += 1
                if not self._serialized_self_attention or (
                    self._serialized_self_attention[-1] != serialized
                ):
                    self._serialized_self_attention.append(serialized)
            if composition is not None:
                serialized = _serialize_payload(
                    composition,
                    "Regional composition diagnostics",
                )
                self._composition_record_count += 1
                if not self._serialized_composition or (
                    self._serialized_composition[-1] != serialized
                ):
                    self._serialized_composition.append(serialized)

    def result(self, run_id: str) -> dict[str, object]:
        """Return an immutable JSON-safe capture result for one workflow run."""

        with self._lock:
            snapshots = [
                cast(dict[str, object], json.loads(value))
                for value in self._serialized_snapshots
            ]
            attention_phases = [
                cast(dict[str, object], json.loads(value))
                for value in self._serialized_attention_phases
            ]
            self_attention = [
                cast(dict[str, object], json.loads(value))
                for value in self._serialized_self_attention
            ]
            composition = [
                cast(dict[str, object], json.loads(value))
                for value in self._serialized_composition
            ]
            return {
                "run_id": run_id,
                "record_count": self._record_count,
                "snapshots": snapshots,
                "attention_phase_record_count": self._attention_phase_record_count,
                "attention_phases": attention_phases,
                "self_attention_record_count": self._self_attention_record_count,
                "self_attention": self_attention,
                "composition_record_count": self._composition_record_count,
                "composition": composition,
            }


def _serialize_payload(value: object, label: str) -> str:
    """Return one validated JSON object without retaining sensitive values."""

    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if not isinstance(json.loads(serialized), dict):
        raise TypeError(f"{label} log payload must be an object.")
    return serialized


@dataclass(frozen=True, slots=True)
class _CaptureState:
    """Retain exact loggers and one handler owned by an active capture."""

    loggers: tuple[logging.Logger, ...]
    handler: _RegionalDiagnosticsHandler


class _LoggerLevelLeaseRegistry:
    """Keep explicit diagnostic loggers at DEBUG while any capture owns them."""

    def __init__(self) -> None:
        """Initialize empty thread-safe logger leases."""

        self._lock = threading.Lock()
        self._leases: dict[str, tuple[int, int]] = {}

    def acquire(self, loggers: tuple[logging.Logger, ...]) -> None:
        """Acquire one DEBUG lease for every target logger."""

        with self._lock:
            for logger in loggers:
                previous = self._leases.get(logger.name)
                if previous is None:
                    self._leases[logger.name] = (logger.level, 1)
                    logger.setLevel(logging.DEBUG)
                    continue
                original_level, owner_count = previous
                self._leases[logger.name] = (original_level, owner_count + 1)

    def release(self, loggers: tuple[logging.Logger, ...]) -> None:
        """Release one lease and restore each logger after its final owner."""

        with self._lock:
            for logger in loggers:
                previous = self._leases.get(logger.name)
                if previous is None:
                    raise RuntimeError(
                        f"Diagnostic logger lease was not acquired: {logger.name!r}."
                    )
                original_level, owner_count = previous
                if owner_count == 1:
                    logger.setLevel(original_level)
                    del self._leases[logger.name]
                else:
                    self._leases[logger.name] = (original_level, owner_count - 1)


_STATES: dict[str, _CaptureState] = {}
_STATE_LOCK = threading.Lock()
_LOGGER_LEVEL_LEASES = _LoggerLevelLeaseRegistry()


class CaptureRegionalDiagnosticsV3(_ComfyNodeBase):
    """Start structured regional diagnostic capture before model execution."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the benchmark model passthrough capture boundary."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.CaptureRegionalDiagnostics",
            display_name="Benchmark Capture Regional Diagnostics",
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
        """Attach one exact logger handler and preserve the supplied model."""

        loggers = tuple(logging.getLogger(name) for name in sorted(_LOGGER_NAMES))
        handler = _RegionalDiagnosticsHandler()
        state = _CaptureState(loggers, handler)
        with _STATE_LOCK:
            if run_id in _STATES:
                raise ValueError(
                    f"Regional diagnostics capture is already active: {run_id!r}."
                )
            _STATES[run_id] = state
        try:
            _LOGGER_LEVEL_LEASES.acquire(loggers)
            for logger in loggers:
                logger.addHandler(handler)
        except BaseException:
            for logger in loggers:
                logger.removeHandler(handler)
            _LOGGER_LEVEL_LEASES.release(loggers)
            with _STATE_LOCK:
                _STATES.pop(run_id, None)
            raise
        return _comfy_io.NodeOutput(model)


class ReadRegionalDiagnosticsV3(_ComfyNodeBase):
    """Finalize and expose one managed regional diagnostic capture."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the latent passthrough and structured JSON output."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.ReadRegionalDiagnostics",
            display_name="Benchmark Read Regional Diagnostics",
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
        """Detach the exact handler and return its complete capture result."""

        with _STATE_LOCK:
            state = _STATES.pop(run_id, None)
        if state is None:
            raise ValueError(
                f"Regional diagnostics capture was not started: {run_id!r}."
            )
        for logger in state.loggers:
            logger.removeHandler(state.handler)
        _LOGGER_LEVEL_LEASES.release(state.loggers)
        result = state.handler.result(run_id)
        if (
            result["record_count"] == 0
            and result["attention_phase_record_count"] == 0
            and result["self_attention_record_count"] == 0
            and result["composition_record_count"] == 0
        ):
            raise ValueError(
                "Regional diagnostics capture did not observe any records."
            )
        diagnostics_json = json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
        return _comfy_io.NodeOutput(
            latent,
            diagnostics_json,
            ui={"regional_diagnostics": [result]},
        )
