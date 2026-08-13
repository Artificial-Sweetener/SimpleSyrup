# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove P9.5 terminal history narrowing and exact acceptance policy."""

from __future__ import annotations

import pytest

from tools.anima_regional_lora_admission_integration.history import (
    RegionalLoraAdmissionError,
    RegionalLoraAdmissionSuccess,
    parse_history,
)
from tools.anima_regional_lora_admission_integration.matrix import (
    RegionalLoraAdmissionCase,
    cases,
)
from tools.anima_regional_lora_admission_integration.validation import (
    validate_history,
)
from tools.anima_regional_lora_admission_integration.workflow import (
    BuiltRegionalLoraAdmissionWorkflow,
    RegionalLoraAdmissionWorkflowBuilder,
)
from tools.comfy_api import JsonObject


def test_success_history_requires_full_surface_single_trajectory_evidence() -> None:
    """Accept eight complete calls and all 448 ADAPTER_A targets."""

    case = cases()[0]
    workflow = _workflow(case)
    observed = parse_history(_success_history(workflow), workflow)

    assert isinstance(observed, RegionalLoraAdmissionSuccess)
    validated = validate_history(case, workflow, observed)
    assert validated.status == "success"
    assert validated.model_call_count == 8
    assert validated.diagnostic_record_count == 8
    assert validated.target_count == 448


def test_error_history_requires_exact_public_sampler_classification() -> None:
    """Accept one target-class failure with no terminal sampler outputs."""

    case = cases()[5]
    workflow = _workflow(case)
    observed = parse_history(_error_history(case, workflow), workflow)

    assert isinstance(observed, RegionalLoraAdmissionError)
    validated = validate_history(case, workflow, observed)
    assert validated.status == "rejected"
    assert validated.model_call_count == 0
    assert validated.exception_type is not None
    assert validated.exception_type.endswith("AnimaRegionalLoraPlanAdmissionError")


def test_mixed_case_requires_second_adapter_identity_and_atomic_failure() -> None:
    """Reject a valid ADAPTER_A followed by one unsupported target without output."""

    case = cases()[-1]
    workflow = _workflow(case)
    observed = parse_history(_error_history(case, workflow), workflow)

    validated = validate_history(case, workflow, observed)
    assert validated.status == "rejected"
    assert validated.target_count == 0


def test_error_parser_rejects_terminal_outputs_or_multiple_errors() -> None:
    """Prevent a sampled or ambiguous failure from satisfying preflight proof."""

    case = cases()[1]
    workflow = _workflow(case)
    history = _error_history(case, workflow)
    outputs = history["outputs"]
    assert isinstance(outputs, dict)
    outputs[workflow.workflow.metrics_node_id] = {"benchmark_metrics": [{}]}
    with pytest.raises(ValueError, match="emitted a terminal sampler output"):
        parse_history(history, workflow)

    history = _error_history(case, workflow)
    status = history["status"]
    assert isinstance(status, dict)
    messages = status["messages"]
    assert isinstance(messages, list)
    messages.append(messages[-1])
    with pytest.raises(ValueError, match="one execution error"):
        parse_history(history, workflow)


def test_validation_rejects_wrong_target_count_and_missing_error_fragment() -> None:
    """Keep success quality and rejection diagnostics exact."""

    success_case = cases()[0]
    success_workflow = _workflow(success_case)
    history = _success_history(success_workflow)
    outputs = _object(history["outputs"])
    diagnostics = _object(outputs[success_workflow.workflow.diagnostics_node_id])
    diagnostic_records = _list(diagnostics["regional_diagnostics"])
    diagnostic = _object(diagnostic_records[0])
    snapshots = _list(diagnostic["snapshots"])
    snapshot = _object(snapshots[0])
    uses = _list(snapshot["adapter_uses"])
    _object(uses[0])["target_count"] = 447
    with pytest.raises(ValueError, match="adapter surface changed"):
        validate_history(
            success_case,
            success_workflow,
            parse_history(history, success_workflow),
        )

    error_case = cases()[2]
    error_workflow = _workflow(error_case)
    observed = parse_history(_error_history(error_case, error_workflow), error_workflow)
    assert isinstance(observed, RegionalLoraAdmissionError)
    changed = RegionalLoraAdmissionError(
        observed.node_id,
        observed.node_type,
        observed.exception_type,
        "different failure",
        observed.executed_node_ids,
    )
    with pytest.raises(ValueError, match="lost an exact classification"):
        validate_history(error_case, error_workflow, changed)


def _workflow(
    case: RegionalLoraAdmissionCase,
) -> BuiltRegionalLoraAdmissionWorkflow:
    """Build one exact synthetic managed graph."""

    return RegionalLoraAdmissionWorkflowBuilder(artifact_phase="p9.5").build(
        case,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )


def _success_history(workflow: BuiltRegionalLoraAdmissionWorkflow) -> JsonObject:
    """Return exact supported metrics and diagnostics."""

    snapshot = {
        "strategy": "attention_coupling",
        "backend": "comfy.ldm.anima.model.Anima",
        "spatial_mode": "full",
        "adapter_uses": [
            {
                "active": True,
                "adapter_token": "token",
                "branch": "positive",
                "composition_index": 0,
                "effective_strength": 0.75,
                "region_index": 0,
                "target_count": 448,
            }
        ],
        "estimated_work": {
            "active_target_count": 448,
            "denoiser_call_multiplier": 1.0,
        },
    }
    return {
        "outputs": {
            workflow.workflow.metrics_node_id: {
                "benchmark_metrics": [
                    {
                        "run_id": workflow.workflow.metrics_run_id,
                        "model_call_count": 8,
                        "runtime_ms": 1.0,
                        "peak_vram_bytes": 1,
                    }
                ]
            },
            workflow.workflow.diagnostics_node_id: {
                "regional_diagnostics": [
                    {
                        "run_id": workflow.workflow.diagnostics_run_id,
                        "record_count": 8,
                        "snapshots": [snapshot.copy() for _ in range(8)],
                    }
                ]
            },
        },
        "status": {"status_str": "success", "messages": []},
    }


def _error_history(
    case: RegionalLoraAdmissionCase,
    workflow: BuiltRegionalLoraAdmissionWorkflow,
) -> JsonObject:
    """Return one exact public-sampler rejection."""

    suffix = case.expected_exception_suffix
    assert suffix is not None
    message = "\n".join(case.expected_error_fragments)
    return {
        "outputs": {},
        "status": {
            "status_str": "error",
            "messages": [
                [
                    "execution_error",
                    {
                        "node_id": workflow.sampler_node_id,
                        "node_type": "SimpleSyrup.KSamplerAttentionCoupling",
                        "exception_type": f"fixture.{suffix}",
                        "exception_message": message,
                        "executed": ["1", "2"],
                    },
                ]
            ],
        },
    }


def _object(value: object) -> JsonObject:
    """Narrow one synthetic JSON object."""

    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return value


def _list(value: object) -> list[object]:
    """Narrow one synthetic JSON list."""

    assert isinstance(value, list)
    return value
