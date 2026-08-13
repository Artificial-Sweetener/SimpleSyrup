# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bind regional-LoRA admission cases to the shared full-context graph."""

from __future__ import annotations

from dataclasses import dataclass

from tools.anima_attention_coupling_workflow import (
    AnimaAttentionCouplingWorkflowBuilder,
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.anima_latent_source_workflow import EmptyAnimaLatentSourceWorkflow

from .conditioning import RegionalLoraAdmissionConditioningWorkflow
from .graph_contract import (
    HEIGHT,
    PUBLIC_NODE_ID,
    STEPS,
    WIDTH,
    RegionalLoraAdmissionGraphCase,
)


@dataclass(frozen=True, slots=True)
class BuiltRegionalLoraAdmissionWorkflow:
    """Retain the shared graph and its unique public sampler identity."""

    workflow: BuiltAnimaAttentionCouplingWorkflow
    sampler_node_id: str
    artifact_phase: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every exact node contract used by the graph."""

        return self.workflow.required_node_ids


class RegionalLoraAdmissionWorkflowBuilder:
    """Build one fully instrumented regional-LoRA admission workflow."""

    def __init__(self, *, artifact_phase: str) -> None:
        """Retain the explicit evidence phase that owns generated artifacts."""

        if not artifact_phase:
            raise ValueError("Admission workflow artifact phase must not be empty.")
        self._artifact_phase = artifact_phase

    def build(
        self,
        case: RegionalLoraAdmissionGraphCase,
        *,
        run_id: str,
        mask_names: tuple[str, ...],
    ) -> BuiltRegionalLoraAdmissionWorkflow:
        """Return one exact eight-step 512-square admission graph."""

        workflow = AnimaAttentionCouplingWorkflowBuilder(
            public_node_id=PUBLIC_NODE_ID,
            artifact_phase=self._artifact_phase,
            conditioning=RegionalLoraAdmissionConditioningWorkflow(case),
            steps=STEPS,
            latent_source=EmptyAnimaLatentSourceWorkflow(WIDTH, HEIGHT),
        ).build(case, run_id=run_id, mask_names=mask_names)
        sampler_ids = tuple(
            node_id
            for node_id, node in workflow.prompt.items()
            if node.get("class_type") == PUBLIC_NODE_ID
        )
        if len(sampler_ids) != 1:
            raise ValueError(
                f"{self._artifact_phase.upper()} workflow must contain one public "
                "sampler node."
            )
        return BuiltRegionalLoraAdmissionWorkflow(
            workflow,
            sampler_ids[0],
            self._artifact_phase,
        )
