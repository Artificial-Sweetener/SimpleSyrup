# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add the benchmark-only pass-through MODEL modifier evidence node."""

from __future__ import annotations

from tools.anima_attention_coupling_workflow import ObservedAnimaModel
from tools.anima_workflow_graph import AnimaWorkflowGraph, NodeReference


class RegionalPatchModelEvidenceWorkflow:
    """Publish upstream MODEL state through one dedicated graph collaborator."""

    def __init__(self, run_id: str) -> None:
        """Retain one stable case-specific evidence identity."""

        if not isinstance(run_id, str) or not run_id:
            raise ValueError("P9.7 model evidence run ID must not be empty.")
        self._run_id = run_id

    def add(
        self,
        graph: AnimaWorkflowGraph,
        *,
        model: NodeReference,
    ) -> ObservedAnimaModel:
        """Return the pass-through snapshot MODEL and evidence node identity."""

        node = graph.add(
            "SimpleSyrupBenchmark.SnapshotModelModifier",
            model=model,
            run_id=self._run_id,
        )
        return ObservedAnimaModel([node, 0], node)
