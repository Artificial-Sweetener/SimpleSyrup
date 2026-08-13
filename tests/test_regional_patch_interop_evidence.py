# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify P9.7 terminal-history decoding and exact evidence validation."""

from __future__ import annotations

import pytest

from tools.comfy_api import ImageReference, JsonObject
from tools.regional_patch_interop_integration.history import (
    RegionalPatchInteropError,
    RegionalPatchInteropHistory,
    RegionalPatchInteropSuccess,
    parse_history,
)
from tools.regional_patch_interop_integration.matrix import (
    STEPS,
    PatchInteropModelFamily,
    PatchInteropModifier,
    PatchInteropSpatialMode,
    RegionalPatchInteropCase,
    cases,
)
from tools.regional_patch_interop_integration.validation import validate_case
from tools.regional_patch_interop_integration.workflow import (
    BuiltRegionalPatchInteropWorkflow,
    RegionalPatchInteropWorkflowBuilder,
)


@pytest.mark.parametrize("case", cases(), ids=lambda case: case.case_id)
def test_every_matrix_case_requires_exact_modifier_and_terminal_evidence(
    case: RegionalPatchInteropCase,
) -> None:
    """Validate the complete accepted/rejected matrix without weaker branches."""

    workflow = _workflow(case)
    snapshot = _modifier_snapshot(case, workflow)
    observed: RegionalPatchInteropHistory
    if case.expect_success:
        model_call_count = STEPS
        observed = RegionalPatchInteropSuccess(
            snapshot,
            {
                "run_id": workflow.metrics_run_id,
                "model_call_count": model_call_count,
                "runtime_ms": 10.0,
                "peak_vram_bytes": 1,
            },
            _diagnostics(case, workflow, record_count=model_call_count),
            ImageReference("result.png", "", "output"),
            None,
        )
    else:
        observed = RegionalPatchInteropError(
            snapshot,
            workflow.sampler_node_id,
            str(workflow.prompt[workflow.sampler_node_id]["class_type"]),
            "builtins.ValueError",
            " | ".join(case.expected_error_fragments),
            (workflow.modifier_snapshot_node_id,),
        )

    validated = validate_case(case, workflow, observed)

    assert validated.status == ("accepted" if case.expect_success else "rejected")
    assert validated.model_call_count == (STEPS if case.expect_success else 0)


def test_success_requires_one_diagnostic_record_per_actual_model_call() -> None:
    """Reject cache evidence whose diagnostics omit an executed model call."""

    case = next(
        item for item in cases() if item.modifier is PatchInteropModifier.EASYCACHE
    )
    workflow = _workflow(case)
    observed = RegionalPatchInteropSuccess(
        _modifier_snapshot(case, workflow),
        {
            "run_id": workflow.metrics_run_id,
            "model_call_count": 5,
            "runtime_ms": 10.0,
            "peak_vram_bytes": 1,
        },
        _diagnostics(case, workflow, record_count=4),
        ImageReference("result.png", "", "output"),
        None,
    )

    with pytest.raises(ValueError, match="one diagnostic record per model call"):
        validate_case(case, workflow, observed)


def test_history_parser_decodes_success_and_rejection_without_crossing_outcomes() -> (
    None
):
    """Require modifier evidence in both outcomes and terminal data only on success."""

    accepted_case = cases()[0]
    accepted_workflow = _workflow(accepted_case)
    accepted_history = _success_history(accepted_case, accepted_workflow)
    accepted = parse_history(accepted_history, accepted_workflow)

    assert isinstance(accepted, RegionalPatchInteropSuccess)
    assert accepted.image_reference.filename == "result.png"

    rejected_case = next(
        case
        for case in cases()
        if case.case_id == "anima-full-scheduled-easycache-rejected"
    )
    rejected_workflow = _workflow(rejected_case)
    rejected_history = _error_history(rejected_case, rejected_workflow)
    rejected = parse_history(rejected_history, rejected_workflow)

    assert isinstance(rejected, RegionalPatchInteropError)
    assert rejected.node_id == rejected_workflow.sampler_node_id
    assert rejected.exception_type == "ValueError"


def test_rejected_history_cannot_publish_terminal_sampler_outputs() -> None:
    """Reject histories that combine an execution error with image/metric evidence."""

    case = next(
        item for item in cases() if item.case_id == "anima-tiled-easycache-rejected"
    )
    workflow = _workflow(case)
    history = _error_history(case, workflow)
    outputs = history["outputs"]
    assert isinstance(outputs, dict)
    outputs[workflow.metrics_node_id] = {"benchmark_metrics": [{}]}

    with pytest.raises(ValueError, match="emitted terminal sampler output"):
        parse_history(history, workflow)


def _workflow(case: RegionalPatchInteropCase) -> BuiltRegionalPatchInteropWorkflow:
    """Build one exact matrix graph for evidence validation."""

    return RegionalPatchInteropWorkflowBuilder().build(
        case,
        run_id="run",
        mask_names=("left.png", "right.png"),
        sdxl_checkpoint_name=(
            "managed.safetensors"
            if case.model_family is PatchInteropModelFamily.SDXL
            else None
        ),
    )


def _modifier_snapshot(
    case: RegionalPatchInteropCase,
    workflow: BuiltRegionalPatchInteropWorkflow,
) -> JsonObject:
    """Return the exact upstream modifier state declared by one case."""

    wrappers: list[object] = []
    cache_type = None
    optimized = None
    object_patches: list[object] = []
    patch_counts: JsonObject = {}
    if case.modifier is PatchInteropModifier.EASYCACHE:
        cache_type = "EasyCacheHolder"
        wrappers = _wrapper_records(
            ("outer_sample", "calc_cond_batch", "diffusion_model"),
            key="easycache",
        )
    elif case.modifier is PatchInteropModifier.LAZYCACHE:
        cache_type = "LazyCacheHolder"
        wrappers = _wrapper_records(
            ("outer_sample", "predict_noise"),
            key="lazycache",
        )
    elif case.modifier is PatchInteropModifier.OPTIMIZED_ATTENTION:
        optimized = "ModelAttentionSelector.execute.<locals>.attention_override"
    elif case.modifier is PatchInteropModifier.NEGPIP:
        patch_counts = {"attn2_patch": 1}
        if case.model_family is PatchInteropModelFamily.ANIMA:
            wrappers = _wrapper_records(("diffusion_model",), key="ppm_negpip_anima")
            object_patches = ["extra_conds"]
    return {
        "run_id": f"{workflow.metrics_run_id}:modifier",
        "cache_holder_type": cache_type,
        "model_function_wrapper": None,
        "optimized_attention_override": optimized,
        "ppm_negpip": case.modifier is PatchInteropModifier.NEGPIP,
        "wrappers": wrappers,
        "object_patch_keys": object_patches,
        "transformer_patch_counts": patch_counts,
    }


def _wrapper_records(types: tuple[str, ...], *, key: str) -> list[object]:
    """Return stable synthetic wrapper records for validation tests."""

    return [
        {"wrapper_type": wrapper_type, "key": key, "callbacks": ["callback"]}
        for wrapper_type in types
    ]


def _diagnostics(
    case: RegionalPatchInteropCase,
    workflow: BuiltRegionalPatchInteropWorkflow,
    *,
    record_count: int = STEPS,
) -> JsonObject:
    """Return exact model-call-aligned diagnostics for one accepted mode."""

    modes = {
        PatchInteropSpatialMode.FULL: ("full",),
        PatchInteropSpatialMode.TILED: ("tile",),
        PatchInteropSpatialMode.CONTEXTUAL: ("tile", "contextual_global"),
    }[case.spatial_mode]
    snapshots = [
        _diagnostic_snapshot(modes[index % len(modes)]) for index in range(record_count)
    ]
    return {
        "run_id": workflow.diagnostics_run_id,
        "record_count": len(snapshots),
        "snapshots": snapshots,
    }


def _diagnostic_snapshot(mode: str) -> JsonObject:
    """Return one exact static-ADAPTER_A diagnostic record."""

    return {
        "strategy": "attention_coupling",
        "backend": "comfy.ldm.anima.model.Anima",
        "spatial_mode": mode,
        "adapter_uses": [
            {
                "active": True,
                "composition_index": 0,
                "region_index": 0,
                "branch": "positive",
                "target_count": 448,
                "effective_strength": 0.75,
                "adapter_token": "adapter_a-token",
            },
            {
                "active": True,
                "composition_index": 1,
                "region_index": 0,
                "branch": "negative",
                "target_count": 448,
                "effective_strength": 0.75,
                "adapter_token": "adapter_a-token",
            },
        ],
        "estimated_work": {
            "active_adapter_uses": 2,
            "active_target_count": 448,
            "target_use_count": 896,
            "denoiser_call_multiplier": 1.0,
        },
    }


def _success_history(
    case: RegionalPatchInteropCase,
    workflow: BuiltRegionalPatchInteropWorkflow,
) -> JsonObject:
    """Return one terminal success history with all required outputs."""

    return {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {
            workflow.modifier_snapshot_node_id: {
                "model_modifier_snapshot": [_modifier_snapshot(case, workflow)]
            },
            workflow.metrics_node_id: {
                "benchmark_metrics": [
                    {
                        "run_id": workflow.metrics_run_id,
                        "model_call_count": STEPS,
                        "runtime_ms": 10.0,
                        "peak_vram_bytes": 1,
                    }
                ]
            },
            workflow.diagnostics_node_id: {
                "regional_diagnostics": [_diagnostics(case, workflow)]
            },
            workflow.save_node_id: {
                "images": [
                    {"filename": "result.png", "subfolder": "", "type": "output"}
                ]
            },
        },
    }


def _error_history(
    case: RegionalPatchInteropCase,
    workflow: BuiltRegionalPatchInteropWorkflow,
) -> JsonObject:
    """Return one terminal named-conflict history without sampler outputs."""

    return {
        "status": {
            "status_str": "error",
            "completed": False,
            "messages": [
                [
                    "execution_error",
                    {
                        "node_id": workflow.sampler_node_id,
                        "node_type": str(
                            workflow.prompt[workflow.sampler_node_id]["class_type"]
                        ),
                        "exception_type": "ValueError",
                        "exception_message": " | ".join(case.expected_error_fragments),
                        "executed": [workflow.modifier_snapshot_node_id],
                    },
                ]
            ],
        },
        "outputs": {
            workflow.modifier_snapshot_node_id: {
                "model_modifier_snapshot": [_modifier_snapshot(case, workflow)]
            }
        },
    }
