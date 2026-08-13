# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify P9.1 accepts exact baseline-derived managed evidence only."""

from __future__ import annotations

from copy import deepcopy

import pytest
from prompt_control_attention_coupling_values import evidence_outputs

from tools.prompt_control_attention_coupling_integration.baseline import load_baseline
from tools.prompt_control_attention_coupling_integration.matrix import cases
from tools.prompt_control_attention_coupling_integration.validation import (
    validate_case_evidence,
)
from tools.prompt_control_attention_coupling_integration.workflow import (
    PromptControlAttentionWorkflowBuilder,
)


def test_every_case_accepts_exact_conditioning_selection_and_adapter_evidence() -> None:
    """Cover text, static, scheduled, stacked, overlap, and inactive behavior."""

    baseline = load_baseline()
    builder = PromptControlAttentionWorkflowBuilder()
    for case in cases():
        workflow = builder.build(
            case,
            run_id="run",
            mask_names=("left.png", "right.png"),
        )
        outputs = evidence_outputs(case, workflow, baseline.observation(case.case_id))

        accepted = validate_case_evidence(
            case,
            workflow,
            outputs,
            baseline.observation(case.case_id),
        )

        assert len(accepted.sampling_sigmas) == 8
        assert accepted.diagnostic_record_count == 8
        assert accepted.conditioning_uuid_count == 16
        assert len(accepted.adapter_tokens) == len(case.adapter_identities)


def test_validation_rejects_extra_calls_and_partial_adapter_targets() -> None:
    """Block per-region trajectories and reduced LoRA target surfaces."""

    baseline = load_baseline()
    case = next(case for case in cases() if case.case_id == "lora-single-scheduled")
    workflow = PromptControlAttentionWorkflowBuilder().build(
        case,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )
    outputs = evidence_outputs(case, workflow, baseline.observation(case.case_id))
    bad_metrics = deepcopy(outputs.metrics)
    bad_metrics["model_call_count"] = 9
    with pytest.raises(ValueError, match="one complete denoiser call"):
        validate_case_evidence(
            case,
            workflow,
            type(outputs)(
                outputs.expansion,
                outputs.snapshot,
                bad_metrics,
                outputs.diagnostics,
            ),
            baseline.observation(case.case_id),
        )

    bad_diagnostics = deepcopy(outputs.diagnostics)
    snapshots = bad_diagnostics["snapshots"]
    assert isinstance(snapshots, list)
    first = snapshots[0]
    assert isinstance(first, dict)
    uses = first["adapter_uses"]
    assert isinstance(uses, list)
    assert isinstance(uses[0], dict)
    uses[0]["target_count"] = 447
    with pytest.raises(ValueError, match="target surface"):
        validate_case_evidence(
            case,
            workflow,
            type(outputs)(
                outputs.expansion,
                outputs.snapshot,
                outputs.metrics,
                bad_diagnostics,
            ),
            baseline.observation(case.case_id),
        )


def test_scheduled_snapshot_accepts_additive_metadata_and_requires_baseline() -> None:
    """Normalize hook references without weakening P0.8 field preservation."""

    baseline = load_baseline()
    case = next(case for case in cases() if case.case_id == "lora-single-scheduled")
    workflow = PromptControlAttentionWorkflowBuilder().build(
        case,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )
    outputs = evidence_outputs(case, workflow, baseline.observation(case.case_id))
    observed = deepcopy(outputs.snapshot)
    hooks = observed["hooks"]
    assert isinstance(hooks, list) and isinstance(hooks[0], dict)
    hooks[0]["hook_ref"] = "pc-stable-public-reference"
    positive = observed["positive"]
    assert isinstance(positive, list) and isinstance(positive[0], dict)
    metadata = positive[0]["metadata"]
    assert isinstance(metadata, dict)
    metadata["attention_mask"] = {"shape": [1, 3], "dtype": "torch.int64"}

    validate_case_evidence(
        case,
        workflow,
        type(outputs)(
            outputs.expansion,
            observed,
            outputs.metrics,
            outputs.diagnostics,
        ),
        baseline.observation(case.case_id),
    )

    metadata.pop("start_percent")
    with pytest.raises(ValueError, match="start_percent is missing"):
        validate_case_evidence(
            case,
            workflow,
            type(outputs)(
                outputs.expansion,
                observed,
                outputs.metrics,
                outputs.diagnostics,
            ),
            baseline.observation(case.case_id),
        )
