"""Validate exact P9.6 non-diffusion pre-sampling rejection evidence."""

from __future__ import annotations

from dataclasses import dataclass

from tools.anima_regional_lora_admission_integration.graph_contract import (
    PUBLIC_NODE_ID,
)
from tools.anima_regional_lora_admission_integration.history import (
    RegionalLoraAdmissionError,
    RegionalLoraAdmissionHistory,
)
from tools.anima_regional_lora_admission_integration.workflow import (
    BuiltRegionalLoraAdmissionWorkflow,
)

from .matrix import AnimaNondiffusionRejectionCase


@dataclass(frozen=True, slots=True)
class ValidatedNondiffusionRejection:
    """Retain normalized terminal owner-rejection evidence."""

    issue_count: int
    exception_type: str
    exception_message: str


def validate_rejection(
    case: AnimaNondiffusionRejectionCase,
    workflow: BuiltRegionalLoraAdmissionWorkflow,
    observed: RegionalLoraAdmissionHistory,
) -> ValidatedNondiffusionRejection:
    """Require one exact public-sampler failure before any model invocation."""

    if workflow.artifact_phase != "p9.6":
        raise ValueError("P9.6 validation requires a P9.6 workflow artifact phase.")
    if not isinstance(observed, RegionalLoraAdmissionError):
        raise ValueError("P9.6 non-diffusion case unexpectedly completed sampling.")
    if observed.node_id != workflow.sampler_node_id:
        raise ValueError("P9.6 rejection did not originate at the public sampler.")
    if observed.node_type != PUBLIC_NODE_ID:
        raise ValueError("P9.6 rejection node type changed.")
    if not observed.exception_type.endswith("AnimaRegionalLoraPlanAdmissionError"):
        raise ValueError("P9.6 rejection exception classification changed.")
    if any(
        fragment not in observed.exception_message
        for fragment in case.expected_error_fragments
    ):
        raise ValueError("P9.6 rejection message lost an owner classification.")
    if workflow.sampler_node_id in observed.executed_node_ids:
        raise ValueError("P9.6 rejected sampler was reported as executed.")
    issue_lines = tuple(
        line
        for line in observed.exception_message.splitlines()
        if line.startswith("- ")
    )
    if len(issue_lines) != case.expected_issue_count:
        raise ValueError("P9.6 rejection issue count changed.")
    expected_prefix = f"- adapter {case.expected_issue_adapter_index} "
    if any(not line.startswith(expected_prefix) for line in issue_lines):
        raise ValueError("P9.6 rejection escaped its expected adapter owner.")
    return ValidatedNondiffusionRejection(
        len(issue_lines),
        observed.exception_type,
        observed.exception_message,
    )
