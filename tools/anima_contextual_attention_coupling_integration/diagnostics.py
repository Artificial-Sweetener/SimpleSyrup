# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate structured P7.8 local and reduced-global diagnostics."""

from __future__ import annotations

from typing import cast

from tools.anima_attention_coupling_workflow import (
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.comfy_api import JsonObject

from .matrix import SUBTOKEN_MASK_CASE_ID, ContextualIntegrationCase


class ContextualDiagnosticsValidator:
    """Own managed Contextual diagnostic identity and semantic validation."""

    @staticmethod
    def validate(
        history: JsonObject,
        workflow: BuiltAnimaAttentionCouplingWorkflow,
        case: ContextualIntegrationCase,
    ) -> JsonObject:
        """Return diagnostics after proving roles, grids, coverage, and survival."""

        outputs = history.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError("P7.8 history is missing outputs.")
        output = outputs.get(workflow.diagnostics_node_id)
        if not isinstance(output, dict):
            raise ValueError("P7.8 history is missing regional diagnostics output.")
        items = output.get("regional_diagnostics")
        if (
            not isinstance(items, list)
            or len(items) != 1
            or not isinstance(items[0], dict)
        ):
            raise ValueError("P7.8 history must contain one diagnostics result.")
        diagnostics = cast(JsonObject, items[0])
        if diagnostics.get("run_id") != workflow.diagnostics_run_id:
            raise ValueError("P7.8 diagnostics identity does not match its workflow.")
        record_count = diagnostics.get("record_count")
        snapshots = diagnostics.get("snapshots")
        if not isinstance(record_count, int) or record_count <= 0:
            raise ValueError("P7.8 diagnostics must report emitted records.")
        if not isinstance(snapshots, list) or not snapshots:
            raise ValueError("P7.8 diagnostics must retain structured snapshots.")
        ContextualDiagnosticsValidator._validate_snapshots(snapshots, case)
        return diagnostics

    @staticmethod
    def _validate_snapshots(
        snapshots: list[object],
        case: ContextualIntegrationCase,
    ) -> None:
        """Prove local/global roles and projected sub-token coverage survival."""

        spatial_modes: set[str] = set()
        surviving_subtoken_global = False
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                raise ValueError("P7.8 diagnostic snapshot must be an object.")
            spatial_mode = snapshot.get("spatial_mode")
            query_grid = snapshot.get("query_grid")
            canonical_mask = snapshot.get("canonical_mask")
            if not isinstance(spatial_mode, str):
                raise ValueError("P7.8 diagnostic snapshot needs a spatial mode.")
            if not isinstance(query_grid, dict) or not isinstance(
                query_grid.get("token_count"), int
            ):
                raise ValueError("P7.8 diagnostics must report projected query grids.")
            if not isinstance(canonical_mask, dict) or not isinstance(
                canonical_mask.get("regions"), list
            ):
                raise ValueError("P7.8 diagnostics must report regional coverage.")
            spatial_modes.add(spatial_mode)
            if spatial_mode == "contextual_global":
                surviving_subtoken_global = (
                    surviving_subtoken_global
                    or ContextualDiagnosticsValidator._region_zero_survives(snapshot)
                )
        ContextualDiagnosticsValidator._validate_roles(spatial_modes, case)
        if case.mask_case_id == SUBTOKEN_MASK_CASE_ID and not surviving_subtoken_global:
            raise ValueError(
                "P7.8 sub-token diagnostics must preserve region zero in the "
                "reduced-global projected plan."
            )

    @staticmethod
    def _region_zero_survives(snapshot: dict[object, object]) -> bool:
        """Return whether reduced-global projection retains region-zero coverage."""

        active_indices = snapshot.get("active_region_indices")
        canonical_mask = snapshot.get("canonical_mask")
        regions = (
            canonical_mask.get("regions") if isinstance(canonical_mask, dict) else None
        )
        return (
            isinstance(active_indices, list)
            and 0 in active_indices
            and isinstance(regions, list)
            and any(
                isinstance(region, dict)
                and region.get("region_index") == 0
                and isinstance(region.get("nonzero_fraction"), int | float)
                and float(region["nonzero_fraction"]) > 0.0
                for region in regions
            )
        )

    @staticmethod
    def _validate_roles(
        spatial_modes: set[str],
        case: ContextualIntegrationCase,
    ) -> None:
        """Require local views and only the declared reduced-global branch."""

        if "tile" not in spatial_modes:
            raise ValueError("P7.8 diagnostics must distinguish local tile views.")
        if case.global_steps > 0 and "contextual_global" not in spatial_modes:
            raise ValueError("P7.8 diagnostics must distinguish reduced-global views.")
        if case.global_steps == 0 and "contextual_global" in spatial_modes:
            raise ValueError("P7.8 local-only diagnostics contain a global view.")
