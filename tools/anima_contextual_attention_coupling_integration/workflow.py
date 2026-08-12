"""Build P7.8 API graphs through the shared Anima workflow owner."""

from __future__ import annotations

from tools.anima_attention_coupling_conditioning import (
    ScheduledRegionalPromptConditioningWorkflow,
)
from tools.anima_attention_coupling_workflow import (
    STEPS,
    AnimaAttentionCouplingWorkflowBuilder,
    AnimaSamplerInputWorkflow,
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.anima_latent_source_workflow import AnimaUpscaleLatentSourceWorkflow
from tools.anima_workflow_graph import AnimaWorkflowGraph

from .matrix import (
    CONTEXT_BATCH_SIZE,
    CONTEXT_OVERLAP,
    CONTEXT_SIZE,
    GLOBAL_DECAY,
    GLOBAL_WEIGHT,
    PUBLIC_NODE_ID,
    REFINEMENT_DENOISE,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    ContextualIntegrationCase,
)


class ContextualSegsSamplerInputWorkflow(AnimaSamplerInputWorkflow):
    """Build one SEGS input from the first authored regional mask."""

    def add(
        self,
        graph: AnimaWorkflowGraph,
        *,
        region_masks: object,
    ) -> dict[str, object]:
        """Return one graph-owned optional SEGS sampler input."""

        mask_images = graph.add(
            "MaskToImage",
            mask=region_masks,
        )
        selected_image = graph.add(
            "ImageFromBatch",
            image=[mask_images, 0],
            batch_index=0,
            length=1,
        )
        selected_mask = graph.add(
            "ImageToMask",
            image=[selected_image, 0],
            channel="red",
        )
        segs = graph.add(
            "SimpleSyrup.MaskToSEGS",
            image=[selected_image, 0],
            mask=[selected_mask, 0],
            mask_threshold=0.5,
            size_threshold=1,
            keep_only=0,
            mask_dilation=0,
            post_dilation=0,
            crop_factor=1.0,
            sort_order="largest to smallest",
            combine_segs=False,
            label="contextual-region",
        )
        return {"segs": [segs, 0]}


class ContextualIntegrationWorkflowBuilder:
    """Bind each matrix case to the shared public-node graph builder."""

    def build(
        self,
        case: ContextualIntegrationCase,
        *,
        run_id: str,
        mask_names: tuple[str, ...],
    ) -> BuiltAnimaAttentionCouplingWorkflow:
        """Return one fully instrumented Contextual public-node graph."""

        conditioning = ScheduledRegionalPromptConditioningWorkflow(
            global_loras=case.global_loras,
            regional_loras=case.regional_loras,
        )
        return AnimaAttentionCouplingWorkflowBuilder(
            public_node_id=PUBLIC_NODE_ID,
            artifact_phase="p7.8",
            conditioning=conditioning,
            steps=STEPS,
            sampler_inputs={
                "diffusion_mode": case.diffusion_mode,
                "latent_context_size": CONTEXT_SIZE,
                "latent_context_overlap": CONTEXT_OVERLAP,
                "latent_context_batch_size": CONTEXT_BATCH_SIZE,
                "global_weight": GLOBAL_WEIGHT,
                "global_steps": case.global_steps,
                "global_decay": GLOBAL_DECAY,
            },
            sampler_input_workflow=(
                ContextualSegsSamplerInputWorkflow() if case.use_segs else None
            ),
            latent_source=AnimaUpscaleLatentSourceWorkflow(
                source_width=SOURCE_WIDTH,
                source_height=SOURCE_HEIGHT,
                target_width=TARGET_WIDTH,
                target_height=TARGET_HEIGHT,
            ),
            denoise=REFINEMENT_DENOISE,
        ).build(case, run_id=run_id, mask_names=mask_names)
