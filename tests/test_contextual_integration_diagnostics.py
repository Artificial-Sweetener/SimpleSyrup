# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify P7.8 managed structured diagnostic validation."""

from __future__ import annotations

import pytest

from tools.anima_contextual_attention_coupling_integration.diagnostics import (
    ContextualDiagnosticsValidator,
)
from tools.anima_contextual_attention_coupling_integration.matrix import cases
from tools.anima_contextual_attention_coupling_integration.workflow import (
    ContextualIntegrationWorkflowBuilder,
)
from tools.comfy_api import JsonObject


def test_validator_accepts_local_and_surviving_subtoken_global_snapshots() -> None:
    """Prove exact roles, query grids, coverage, and sub-token adapter survival."""

    case = cases()[2]
    workflow = ContextualIntegrationWorkflowBuilder().build(
        case,
        run_id="run",
        mask_names=("pixel.png", "rest.png"),
    )
    global_snapshot = _snapshot("contextual_global")
    adapter_uses = global_snapshot["adapter_uses"]
    assert isinstance(adapter_uses, list)
    assert isinstance(adapter_uses[0], dict)
    adapter_uses[0]["active"] = False
    history = _history(
        workflow.diagnostics_node_id,
        workflow.diagnostics_run_id,
        [_snapshot("tile"), global_snapshot],
    )

    result = ContextualDiagnosticsValidator.validate(history, workflow, case)

    assert result["record_count"] == 2


def test_validator_rejects_missing_declared_reduced_global_role() -> None:
    """Fail when a reduced-global workflow only reports local tile snapshots."""

    case = cases()[1]
    workflow = ContextualIntegrationWorkflowBuilder().build(
        case,
        run_id="run",
        mask_names=("left.png", "right.png"),
    )
    history = _history(
        workflow.diagnostics_node_id,
        workflow.diagnostics_run_id,
        [_snapshot("tile")],
    )

    with pytest.raises(ValueError, match="distinguish reduced-global"):
        ContextualDiagnosticsValidator.validate(history, workflow, case)


def test_validator_rejects_missing_subtoken_global_region() -> None:
    """Fail when the one-pixel region disappears from reduced-global projection."""

    case = cases()[2]
    workflow = ContextualIntegrationWorkflowBuilder().build(
        case,
        run_id="run",
        mask_names=("pixel.png", "rest.png"),
    )
    global_snapshot = _snapshot("contextual_global")
    global_snapshot["active_region_indices"] = [1]
    history = _history(
        workflow.diagnostics_node_id,
        workflow.diagnostics_run_id,
        [_snapshot("tile"), global_snapshot],
    )

    with pytest.raises(ValueError, match="preserve region zero"):
        ContextualDiagnosticsValidator.validate(history, workflow, case)


def _history(
    node_id: str,
    run_id: str,
    snapshots: list[object],
) -> JsonObject:
    """Return one managed-history diagnostics output."""

    return {
        "outputs": {
            node_id: {
                "regional_diagnostics": [
                    {
                        "run_id": run_id,
                        "record_count": len(snapshots),
                        "snapshots": snapshots,
                    }
                ]
            }
        }
    }


def _snapshot(spatial_mode: str) -> JsonObject:
    """Return one complete structured diagnostic snapshot."""

    return {
        "spatial_mode": spatial_mode,
        "active_region_indices": [0, 1],
        "canonical_mask": {
            "regions": [
                {"region_index": 0, "nonzero_fraction": 0.000651},
                {"region_index": 1, "nonzero_fraction": 0.999349},
            ]
        },
        "query_grid": {"token_count": 36864},
        "adapter_uses": [
            {"region_index": 0, "active": True},
            {"region_index": 1, "active": True},
        ],
    }
