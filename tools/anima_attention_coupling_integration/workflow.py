"""Bind P5.9 cases to the shared Anima workflow graph owner."""

from __future__ import annotations

from tools.anima_attention_coupling_conditioning import (
    ScheduledRegionalPromptConditioningWorkflow,
)
from tools.anima_attention_coupling_workflow import (
    STEPS,
    AnimaAttentionCouplingWorkflowBuilder,
    BuiltAnimaAttentionCouplingWorkflow,
)

from .matrix import PUBLIC_NODE_ID, IntegrationCase


class IntegrationWorkflowBuilder:
    """Build P5.9 graphs with the accepted scheduled conditioning policy."""

    def build(
        self,
        case: IntegrationCase,
        *,
        run_id: str,
        mask_names: tuple[str, ...],
    ) -> BuiltAnimaAttentionCouplingWorkflow:
        """Return one fully instrumented public-node workflow."""

        conditioning = ScheduledRegionalPromptConditioningWorkflow(
            global_loras=case.global_loras,
            regional_loras=case.regional_loras,
        )
        return AnimaAttentionCouplingWorkflowBuilder(
            public_node_id=PUBLIC_NODE_ID,
            artifact_phase="p5.9",
            conditioning=conditioning,
            steps=STEPS,
        ).build(case, run_id=run_id, mask_names=mask_names)
