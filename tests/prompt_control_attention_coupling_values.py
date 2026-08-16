# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build deterministic P9.1 evidence values from the validated P0.8 baseline."""

from __future__ import annotations

from copy import deepcopy
from uuid import UUID

from tools.anima_attention_coupling_workflow import (
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.comfy_api import JsonObject
from tools.prompt_control_attention_coupling_integration.baseline import (
    PromptControlBaselineObservation,
)
from tools.prompt_control_attention_coupling_integration.history import (
    PromptControlAttentionOutputs,
)
from tools.prompt_control_attention_coupling_integration.matrix import (
    PromptControlAttentionCase,
)


def evidence_outputs(
    case: PromptControlAttentionCase,
    workflow: BuiltAnimaAttentionCouplingWorkflow,
    baseline: PromptControlBaselineObservation,
) -> PromptControlAttentionOutputs:
    """Return one complete baseline-derived managed evidence fixture."""

    return PromptControlAttentionOutputs(
        deepcopy(baseline.expansion),
        _snapshot(case, baseline.snapshot),
        {
            "run_id": workflow.metrics_run_id,
            "runtime_ms": 100.0,
            "model_call_count": 8,
            "peak_vram_bytes": 1024,
        },
        _diagnostics(case, workflow, baseline.runtime),
    )


def _snapshot(case: PromptControlAttentionCase, baseline: JsonObject) -> JsonObject:
    """Add only the expected static regional hook difference."""

    result = deepcopy(baseline)
    if not case.characterization.expect_static_model_lora:
        return result
    hook: JsonObject = {
        "identity": "primary_adapter-static-0.75",
        "order": 0,
        "hook_type": "WeightHook",
        "hook_ref": "opaque-hook-ref-0",
        "hook_id": None,
        "hook_scope": "hooked_only",
        "base_strength_model": 0.75,
        "base_strength_clip": 0.25,
        "effective_strength_model": 0.75,
        "effective_strength_clip": 0.25,
        "keyframes": [{"start_percent": 0.0, "strength": 1.0, "guarantee_steps": 1}],
    }
    result["hooks"] = [hook]
    for side in ("positive", "negative"):
        entries = result[side]
        assert isinstance(entries, list)
        for raw in entries:
            assert isinstance(raw, dict)
            metadata = raw["metadata"]
            assert isinstance(metadata, dict)
            metadata["hooks"] = {"hook_group": [deepcopy(hook)]}
    return result


def _diagnostics(
    case: PromptControlAttentionCase,
    workflow: BuiltAnimaAttentionCouplingWorkflow,
    runtime: JsonObject,
) -> JsonObject:
    """Build one snapshot per raw P0.8 timestep without schedule evaluation."""

    grouped = _group_calls(runtime)
    snapshots = [
        _diagnostic_snapshot(case, index, timestep, values)
        for index, (timestep, values) in enumerate(grouped.items(), start=1)
    ]
    return {
        "run_id": workflow.diagnostics_run_id,
        "record_count": len(snapshots),
        "snapshots": snapshots,
    }


def _group_calls(runtime: JsonObject) -> dict[float, JsonObject]:
    """Aggregate exact baseline calls by their observed timestep."""

    calls = runtime["calls"]
    assert isinstance(calls, list)
    grouped: dict[float, JsonObject] = {}
    for raw in calls:
        assert isinstance(raw, dict)
        timestep = float(raw["timestep"])
        strengths = list(raw["effective_strengths"])
        values = grouped.setdefault(
            timestep,
            {"positive_count": 0, "negative_count": 0, "strengths": strengths},
        )
        selectors = raw["cond_or_uncond"]
        assert isinstance(selectors, list)
        for selector in selectors:
            field = "positive_count" if selector == 0 else "negative_count"
            count = values[field]
            assert isinstance(count, int)
            values[field] = count + 1
    return grouped


def _diagnostic_snapshot(
    case: PromptControlAttentionCase,
    index: int,
    timestep: float,
    expected: JsonObject,
) -> JsonObject:
    """Return one full-context Anima diagnostic fixture."""

    chunks = [
        {"chunk_index": 0, "branch": "positive", "batch_start": 0, "batch_stop": 1},
        {"chunk_index": 1, "branch": "negative", "batch_start": 1, "batch_stop": 2},
    ]
    positive_count_value = expected["positive_count"]
    negative_count_value = expected["negative_count"]
    assert isinstance(positive_count_value, int)
    assert isinstance(negative_count_value, int)
    positive_count = positive_count_value
    negative_count = negative_count_value
    count = max(positive_count, negative_count, 1)
    positive_strength = next(
        (
            item.strength
            for item in case.characterization.expected_positive
            if item.strength is not None
        ),
        1.0,
    )
    entries: list[object] = [
        {
            "region_index": 0,
            "entry_index": entry_index,
            "strengths": [
                positive_strength if entry_index < positive_count else 0.0,
                1.0 if entry_index < negative_count else 0.0,
            ],
            "active": entry_index < max(positive_count, negative_count),
        }
        for entry_index in range(count)
    ]
    entries.append(
        {
            "region_index": 1,
            "entry_index": 0,
            "strengths": [1.0, 1.0],
            "active": True,
        }
    )
    first_uuid = str(UUID(int=index * 2 - 1, version=4))
    second_uuid = str(UUID(int=index * 2, version=4))
    snapshot: JsonObject = {
        "model_call": {
            "sampling_sigma": timestep,
            "conditioning_uuids": [first_uuid, second_uuid],
        },
        "batch_layout": {"chunks": chunks},
        "regional_conditioning_entries": entries,
        "estimated_work": {"denoiser_call_multiplier": 1.0},
    }
    if case.has_regional_hooks:
        snapshot["adapter_uses"] = _adapter_uses(case, expected)
    return snapshot


def _adapter_uses(
    case: PromptControlAttentionCase,
    expected: JsonObject,
) -> list[object]:
    """Return exact positive/negative full-target adapter uses."""

    if not case.has_regional_hooks:
        return []
    if case.characterization.expect_static_model_lora:
        strengths = [0.75]
    else:
        strengths_value = expected["strengths"]
        assert isinstance(strengths_value, list)
        assert all(isinstance(value, int | float) for value in strengths_value)
        strengths = [float(value) for value in strengths_value]
    uses: list[object] = []
    for branch_index, branch in enumerate(("positive", "negative")):
        for adapter_index, strength in enumerate(strengths):
            uses.append(
                {
                    "composition_index": branch_index * len(strengths) + adapter_index,
                    "adapter_token": f"token{adapter_index:011d}",
                    "region_index": 0,
                    "branch": branch,
                    "target_count": 448,
                    "effective_strength": strength,
                    "active": strength != 0.0,
                    "pruning_reason": None if strength != 0.0 else "schedule_zero",
                }
            )
    return uses
