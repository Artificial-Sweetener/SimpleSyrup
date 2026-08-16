# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate exact P9.5 supported execution and fail-closed classifications."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tools.comfy_api import JsonObject

from .graph_contract import STEPS
from .history import (
    RegionalLoraAdmissionError,
    RegionalLoraAdmissionHistory,
    RegionalLoraAdmissionSuccess,
)
from .matrix import RegionalLoraAdmissionCase
from .workflow import BuiltRegionalLoraAdmissionWorkflow


@dataclass(frozen=True, slots=True)
class ValidatedRegionalLoraAdmission:
    """Retain normalized terminal evidence for durable publication."""

    status: str
    model_call_count: int
    diagnostic_record_count: int
    target_count: int
    exception_type: str | None = None
    exception_message: str | None = None


def validate_history(
    case: RegionalLoraAdmissionCase,
    workflow: BuiltRegionalLoraAdmissionWorkflow,
    observed: RegionalLoraAdmissionHistory,
) -> ValidatedRegionalLoraAdmission:
    """Require the case's exact success or pre-sampling rejection contract."""

    if case.expect_success:
        if not isinstance(observed, RegionalLoraAdmissionSuccess):
            raise ValueError(
                "P9.5 supported PRIMARY_ADAPTER case did not complete successfully."
            )
        return _validate_success(workflow, observed)
    if not isinstance(observed, RegionalLoraAdmissionError):
        raise ValueError("P9.5 rejection case unexpectedly completed sampling.")
    return _validate_rejection(case, workflow, observed)


def _validate_success(
    workflow: BuiltRegionalLoraAdmissionWorkflow,
    observed: RegionalLoraAdmissionSuccess,
) -> ValidatedRegionalLoraAdmission:
    """Require one exact full-surface, single-trajectory PRIMARY_ADAPTER execution."""

    metrics = observed.metrics
    if metrics.get("run_id") != workflow.workflow.metrics_run_id:
        raise ValueError("P9.5 metrics identity changed.")
    if metrics.get("model_call_count") != STEPS:
        raise ValueError(
            "P9.5 supported PRIMARY_ADAPTER must use one model call per step."
        )
    if _number(metrics.get("runtime_ms"), "runtime_ms") <= 0.0:
        raise ValueError("P9.5 instrumented runtime must be positive.")
    peak = metrics.get("peak_vram_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise ValueError("P9.5 peak VRAM must be a non-negative integer.")

    diagnostics = observed.diagnostics
    if diagnostics.get("run_id") != workflow.workflow.diagnostics_run_id:
        raise ValueError("P9.5 diagnostic identity changed.")
    snapshots = _array(diagnostics.get("snapshots"), "diagnostic snapshots")
    if diagnostics.get("record_count") != STEPS or len(snapshots) != STEPS:
        raise ValueError("P9.5 diagnostics must cover every model call.")
    adapter_tokens: set[str] = set()
    for raw_snapshot in snapshots:
        snapshot = _object(raw_snapshot, "diagnostic snapshot")
        if (
            snapshot.get("strategy") != "attention_coupling"
            or snapshot.get("backend") != "comfy.ldm.anima.model.Anima"
            or snapshot.get("spatial_mode") != "full"
        ):
            raise ValueError("P9.5 diagnostic backend or strategy changed.")
        uses = _array(snapshot.get("adapter_uses"), "adapter_uses")
        if len(uses) != 1:
            raise ValueError(
                "P9.5 supported PRIMARY_ADAPTER must expose one regional use."
            )
        use = _object(uses[0], "adapter use")
        if (
            use.get("active") is not True
            or use.get("branch") != "positive"
            or use.get("composition_index") != 0
            or use.get("region_index") != 0
            or use.get("target_count") != 448
        ):
            raise ValueError("P9.5 supported PRIMARY_ADAPTER adapter surface changed.")
        _same_number(use.get("effective_strength"), 0.75, "adapter strength")
        token = use.get("adapter_token")
        if not isinstance(token, str) or not token:
            raise ValueError("P9.5 supported PRIMARY_ADAPTER adapter token is missing.")
        adapter_tokens.add(token)
        work = _object(snapshot.get("estimated_work"), "estimated_work")
        if work.get("active_target_count") != 448:
            raise ValueError("P9.5 active target count changed.")
        _same_number(
            work.get("denoiser_call_multiplier"),
            1.0,
            "denoiser multiplier",
        )
    if len(adapter_tokens) != 1:
        raise ValueError("P9.5 adapter identity changed during sampling.")
    return ValidatedRegionalLoraAdmission("success", STEPS, STEPS, 448)


def _validate_rejection(
    case: RegionalLoraAdmissionCase,
    workflow: BuiltRegionalLoraAdmissionWorkflow,
    observed: RegionalLoraAdmissionError,
) -> ValidatedRegionalLoraAdmission:
    """Require one exact public-sampler failure before any model invocation."""

    if observed.node_id != workflow.sampler_node_id:
        raise ValueError("P9.5 rejection did not originate at the public sampler.")
    if observed.node_type != "SimpleSyrup.KSamplerAttentionCoupling":
        raise ValueError("P9.5 rejection node type changed.")
    expected_suffix = case.expected_exception_suffix
    if expected_suffix is None or not observed.exception_type.endswith(expected_suffix):
        raise ValueError("P9.5 rejection exception classification changed.")
    if any(
        fragment not in observed.exception_message
        for fragment in case.expected_error_fragments
    ):
        raise ValueError("P9.5 rejection message lost an exact classification.")
    if workflow.sampler_node_id in observed.executed_node_ids:
        raise ValueError("P9.5 rejected sampler was reported as executed.")
    return ValidatedRegionalLoraAdmission(
        "rejected",
        0,
        0,
        0,
        observed.exception_type,
        observed.exception_message,
    )


def _same_number(value: object, expected: float, field: str) -> None:
    """Require one finite number within exact evidence tolerance."""

    if not math.isclose(_number(value, field), expected, abs_tol=1e-8):
        raise ValueError(f"P9.5 {field} changed.")


def _number(value: object, field: str) -> float:
    """Narrow one finite JSON number."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise TypeError(f"P9.5 {field} must be finite numeric data.")
    return float(value)


def _array(value: object, field: str) -> list[object]:
    """Narrow one JSON list."""

    if not isinstance(value, list):
        raise TypeError(f"P9.5 {field} must be a list.")
    return value


def _object(value: object, field: str) -> JsonObject:
    """Narrow one JSON object with string keys."""

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"P9.5 {field} must be an object.")
    return value
