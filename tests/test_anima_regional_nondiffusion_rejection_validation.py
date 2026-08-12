"""Prove exact P9.6 history narrowing and owner-rejection policy."""

from __future__ import annotations

import pytest
from anima_nondiffusion_rejection_values import synthetic_rejection_history

from tools.anima_regional_lora_admission_integration.history import (
    RegionalLoraAdmissionError,
    parse_history,
)
from tools.anima_regional_lora_admission_integration.workflow import (
    RegionalLoraAdmissionWorkflowBuilder,
)
from tools.anima_regional_nondiffusion_rejection_integration.matrix import cases
from tools.anima_regional_nondiffusion_rejection_integration.validation import (
    validate_rejection,
)


def test_every_case_rejects_with_exact_issue_count_and_adapter_owner() -> None:
    """Require 60, one, two, and adapter-one aggregate issue boundaries."""

    for case in cases():
        workflow = RegionalLoraAdmissionWorkflowBuilder(artifact_phase="p9.6").build(
            case,
            run_id="run",
            mask_names=("left.png", "right.png"),
        )
        observed = parse_history(
            synthetic_rejection_history(case, workflow),
            workflow,
        )

        assert isinstance(observed, RegionalLoraAdmissionError)
        validated = validate_rejection(case, workflow, observed)
        assert validated.issue_count == case.expected_issue_count


def test_validation_rejects_partial_issue_sets_and_executed_sampler() -> None:
    """Prevent incomplete admission evidence or sampling from passing P9.6."""

    case = cases()[0]
    workflow = RegionalLoraAdmissionWorkflowBuilder(artifact_phase="p9.6").build(
        case,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )
    observed = parse_history(synthetic_rejection_history(case, workflow), workflow)
    assert isinstance(observed, RegionalLoraAdmissionError)
    lines = observed.exception_message.splitlines()
    incomplete = RegionalLoraAdmissionError(
        observed.node_id,
        observed.node_type,
        observed.exception_type,
        "\n".join(lines[:-1]),
        observed.executed_node_ids,
    )
    with pytest.raises(ValueError, match="owner classification|issue count"):
        validate_rejection(case, workflow, incomplete)

    executed = RegionalLoraAdmissionError(
        observed.node_id,
        observed.node_type,
        observed.exception_type,
        observed.exception_message,
        (*observed.executed_node_ids, workflow.sampler_node_id),
    )
    with pytest.raises(ValueError, match="reported as executed"):
        validate_rejection(case, workflow, executed)
