# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bind P9.4 conditioning to full, tiled, and Contextual public samplers."""

from __future__ import annotations

from dataclasses import replace

from tools.anima_attention_coupling_integration.matrix import (
    PUBLIC_NODE_ID as FULL_NODE_ID,
)
from tools.anima_attention_coupling_workflow import (
    STEPS,
    AnimaAttentionCouplingWorkflowBuilder,
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.anima_contextual_attention_coupling_integration.matrix import (
    CONTEXT_BATCH_SIZE,
    CONTEXT_OVERLAP,
    CONTEXT_SIZE,
    GLOBAL_DECAY,
    GLOBAL_STEPS,
    GLOBAL_WEIGHT,
    REFINEMENT_DENOISE,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)
from tools.anima_contextual_attention_coupling_integration.matrix import (
    PUBLIC_NODE_ID as CONTEXTUAL_NODE_ID,
)
from tools.anima_latent_source_workflow import AnimaUpscaleLatentSourceWorkflow
from tools.anima_tiled_attention_coupling_integration.matrix import (
    PUBLIC_NODE_ID as TILED_NODE_ID,
)
from tools.anima_tiled_attention_coupling_integration.matrix import (
    TILE_HEIGHT,
    TILE_OVERLAP,
    TILE_WIDTH,
)

from .conditioning import TextEncoderLoraConditioningWorkflow
from .fixture import TextEncoderLoraFixtureIdentity
from .matrix import TextEncoderLoraCase, TextEncoderLoraSpatialMode


class TextEncoderLoraWorkflowBuilder:
    """Build one instrumented P9.4 workflow for its declared spatial mode."""

    def __init__(self, fixture: TextEncoderLoraFixtureIdentity) -> None:
        """Retain the externally selected text-encoder adapter identity."""

        self._fixture = fixture

    def build(
        self,
        case: TextEncoderLoraCase,
        *,
        run_id: str,
        mask_names: tuple[str, ...],
    ) -> BuiltAnimaAttentionCouplingWorkflow:
        """Bind one case to the accepted shared Anima graph owner."""

        conditioning = TextEncoderLoraConditioningWorkflow(case, self._fixture)
        if case.spatial_mode is TextEncoderLoraSpatialMode.FULL:
            builder = AnimaAttentionCouplingWorkflowBuilder(
                public_node_id=FULL_NODE_ID,
                artifact_phase="p9.4",
                conditioning=conditioning,
                steps=STEPS,
            )
        else:
            latent_source = AnimaUpscaleLatentSourceWorkflow(
                source_width=SOURCE_WIDTH,
                source_height=SOURCE_HEIGHT,
                target_width=TARGET_WIDTH,
                target_height=TARGET_HEIGHT,
            )
            if case.spatial_mode is TextEncoderLoraSpatialMode.TILED:
                builder = AnimaAttentionCouplingWorkflowBuilder(
                    public_node_id=TILED_NODE_ID,
                    artifact_phase="p9.4",
                    conditioning=conditioning,
                    steps=STEPS,
                    sampler_inputs={
                        "diffusion_mode": "multidiffusion",
                        "latent_tile_width": TILE_WIDTH,
                        "latent_tile_height": TILE_HEIGHT,
                        "latent_tile_overlap": TILE_OVERLAP,
                        "latent_tile_batch_size": 4,
                    },
                    latent_source=latent_source,
                    denoise=REFINEMENT_DENOISE,
                )
            else:
                builder = AnimaAttentionCouplingWorkflowBuilder(
                    public_node_id=CONTEXTUAL_NODE_ID,
                    artifact_phase="p9.4",
                    conditioning=conditioning,
                    steps=STEPS,
                    sampler_inputs={
                        "diffusion_mode": "multidiffusion",
                        "latent_context_size": CONTEXT_SIZE,
                        "latent_context_overlap": CONTEXT_OVERLAP,
                        "latent_context_batch_size": CONTEXT_BATCH_SIZE,
                        "global_weight": GLOBAL_WEIGHT,
                        "global_steps": GLOBAL_STEPS,
                        "global_decay": GLOBAL_DECAY,
                    },
                    latent_source=latent_source,
                    denoise=REFINEMENT_DENOISE,
                )
        built = builder.build(case, run_id=run_id, mask_names=mask_names)
        return self._with_conditioning_snapshot(built, case=case, run_id=run_id)

    @staticmethod
    def _with_conditioning_snapshot(
        workflow: BuiltAnimaAttentionCouplingWorkflow,
        *,
        case: TextEncoderLoraCase,
        run_id: str,
    ) -> BuiltAnimaAttentionCouplingWorkflow:
        """Append one output-only snapshot of the public encoder's exact batches."""

        encoder_node_ids = [
            node_id
            for node_id, node in workflow.prompt.items()
            if node.get("class_type")
            == "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl"
        ]
        if len(encoder_node_ids) != 1:
            raise ValueError(
                "P9.4 workflow requires one Schedule-and-Encode conditioning owner."
            )
        prompt = workflow.prompt.copy()
        snapshot_node_id = str(max(int(node_id) for node_id in prompt) + 1)
        snapshot_run_id = f"{run_id}:{case.case_id}:conditioning-batch"
        prompt[snapshot_node_id] = {
            "class_type": "SimpleSyrupBenchmark.SnapshotConditioningBatch",
            "inputs": {
                "positive": [encoder_node_ids[0], 1],
                "negative": [encoder_node_ids[0], 2],
                "run_id": snapshot_run_id,
            },
        }
        return replace(
            workflow,
            prompt=prompt,
            conditioning_batch_snapshot_node_id=snapshot_node_id,
            conditioning_batch_snapshot_run_id=snapshot_run_id,
        )
