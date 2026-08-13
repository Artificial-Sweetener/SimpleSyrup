# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bind P9.1 conditioning to the shared full-context sampler graph."""

from __future__ import annotations

from tools.anima_attention_coupling_workflow import (
    AnimaAttentionCouplingWorkflowBuilder,
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.anima_latent_source_workflow import EmptyAnimaLatentSourceWorkflow

from .conditioning import PromptControlAttentionConditioningWorkflow
from .matrix import HEIGHT, PUBLIC_NODE_ID, STEPS, WIDTH, PromptControlAttentionCase


class PromptControlAttentionWorkflowBuilder:
    """Build fully instrumented P9.1 public-node workflows."""

    def build(
        self,
        case: PromptControlAttentionCase,
        *,
        run_id: str,
        mask_names: tuple[str, ...],
    ) -> BuiltAnimaAttentionCouplingWorkflow:
        """Return one exact eight-step 512x512 managed graph."""

        return AnimaAttentionCouplingWorkflowBuilder(
            public_node_id=PUBLIC_NODE_ID,
            artifact_phase="p9.1",
            conditioning=PromptControlAttentionConditioningWorkflow(case),
            steps=STEPS,
            latent_source=EmptyAnimaLatentSourceWorkflow(WIDTH, HEIGHT),
        ).build(case, run_id=run_id, mask_names=mask_names)
