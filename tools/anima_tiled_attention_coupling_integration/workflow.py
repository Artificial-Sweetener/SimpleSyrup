"""Build P6.9 tiled API graphs through the shared Anima workflow owner."""

from __future__ import annotations

from tools.anima_attention_coupling_conditioning import (
    ScheduledRegionalPromptConditioningWorkflow,
)
from tools.anima_attention_coupling_workflow import (
    STEPS,
    AnimaAttentionCouplingWorkflowBuilder,
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.anima_latent_source_workflow import AnimaUpscaleLatentSourceWorkflow

from .matrix import (
    PUBLIC_NODE_ID,
    REFINEMENT_DENOISE,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    TILE_HEIGHT,
    TILE_OVERLAP,
    TILE_WIDTH,
    TiledIntegrationCase,
)


class TiledIntegrationWorkflowBuilder:
    """Bind each matrix case's tiled controls to the shared graph builder."""

    def build(
        self,
        case: TiledIntegrationCase,
        *,
        run_id: str,
        mask_names: tuple[str, ...],
    ) -> BuiltAnimaAttentionCouplingWorkflow:
        """Return one fully instrumented public tiled-node graph."""

        conditioning = ScheduledRegionalPromptConditioningWorkflow(
            global_loras=case.global_loras,
            regional_loras=case.regional_loras,
        )
        return AnimaAttentionCouplingWorkflowBuilder(
            public_node_id=PUBLIC_NODE_ID,
            artifact_phase="p6.9",
            conditioning=conditioning,
            steps=STEPS,
            sampler_inputs={
                "diffusion_mode": case.diffusion_mode,
                "latent_tile_width": TILE_WIDTH,
                "latent_tile_height": TILE_HEIGHT,
                "latent_tile_overlap": TILE_OVERLAP,
                "latent_tile_batch_size": case.tile_batch_size,
            },
            latent_source=AnimaUpscaleLatentSourceWorkflow(
                source_width=SOURCE_WIDTH,
                source_height=SOURCE_HEIGHT,
                target_width=TARGET_WIDTH,
                target_height=TARGET_HEIGHT,
            ),
            denoise=REFINEMENT_DENOISE,
        ).build(case, run_id=run_id, mask_names=mask_names)
