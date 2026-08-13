# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build the public managed P9.7 accepted and rejected workflow graphs."""

from __future__ import annotations

from dataclasses import dataclass

from tools.anima_attention_coupling_conditioning import (
    ScheduledRegionalPromptConditioningWorkflow,
)
from tools.anima_attention_coupling_workflow import (
    AnimaAttentionCouplingWorkflowBuilder,
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.anima_contextual_attention_coupling_integration.matrix import (
    CONTEXT_BATCH_SIZE,
    CONTEXT_OVERLAP,
    CONTEXT_SIZE,
    GLOBAL_DECAY,
    GLOBAL_WEIGHT,
)
from tools.anima_latent_source_workflow import (
    AnimaLatentSourceWorkflow,
    AnimaUpscaleLatentSourceWorkflow,
    EmptyAnimaLatentSourceWorkflow,
)
from tools.anima_tiled_attention_coupling_integration.matrix import (
    REFINEMENT_DENOISE,
    TILE_HEIGHT,
    TILE_OVERLAP,
    TILE_WIDTH,
)
from tools.anima_workflow_graph import AnimaWorkflowGraph, NodeReference
from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.matrix import (
    NEGATIVE_PROMPTS as SDXL_NEGATIVE_PROMPTS,
)
from tools.sdxl_attention_coupling_integration.matrix import (
    POSITIVE_PROMPTS as SDXL_POSITIVE_PROMPTS,
)

from .latent import UnsampledAnimaUpscaleLatentSourceWorkflow
from .matrix import (
    FULL_HEIGHT,
    FULL_WIDTH,
    STEPS,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    PatchInteropModelFamily,
    PatchInteropSpatialMode,
    RegionalPatchInteropCase,
)
from .model_evidence import RegionalPatchModelEvidenceWorkflow
from .modifier import RegionalPatchModifierWorkflow

FULL_NODE_ID = "SimpleSyrup.KSamplerAttentionCoupling"
TILED_NODE_ID = "SimpleSyrup.KSamplerAttentionCouplingTiled"
CONTEXTUAL_NODE_ID = "SimpleSyrup.KSamplerAttentionCouplingContextual"


@dataclass(frozen=True, slots=True)
class BuiltRegionalPatchInteropWorkflow:
    """Retain one graph and every exact terminal evidence node identity."""

    prompt: dict[str, JsonObject]
    sampler_node_id: str
    modifier_snapshot_node_id: str
    save_node_id: str
    metrics_node_id: str
    diagnostics_node_id: str
    metrics_run_id: str
    diagnostics_run_id: str
    source_save_node_id: str | None = None

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every exact class required from managed node discovery."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


class RegionalPatchInteropWorkflowBuilder:
    """Build one model-family-specific P9.7 public-sampler graph."""

    def build(
        self,
        case: RegionalPatchInteropCase,
        *,
        run_id: str,
        mask_names: tuple[str, ...],
        sdxl_checkpoint_name: str | None = None,
    ) -> BuiltRegionalPatchInteropWorkflow:
        """Return one exact graph with modifier and terminal evidence nodes."""

        if case.model_family is PatchInteropModelFamily.ANIMA:
            return self._build_anima(case, run_id=run_id, mask_names=mask_names)
        if sdxl_checkpoint_name is None:
            raise ValueError("P9.7 SDXL workflow requires a checkpoint name.")
        return self._build_sdxl(
            case,
            run_id=run_id,
            mask_names=mask_names,
            checkpoint_name=sdxl_checkpoint_name,
        )

    @staticmethod
    def _build_anima(
        case: RegionalPatchInteropCase,
        *,
        run_id: str,
        mask_names: tuple[str, ...],
    ) -> BuiltRegionalPatchInteropWorkflow:
        """Build one shared Anima graph with explicit modifier and observer."""

        spatial = case.spatial_mode
        latent_source: AnimaLatentSourceWorkflow
        if spatial is PatchInteropSpatialMode.FULL:
            node_id = FULL_NODE_ID
            sampler_inputs: dict[str, object] = {}
            latent_source = EmptyAnimaLatentSourceWorkflow(FULL_WIDTH, FULL_HEIGHT)
            denoise = 1.0
        else:
            node_id = (
                TILED_NODE_ID
                if spatial is PatchInteropSpatialMode.TILED
                else CONTEXTUAL_NODE_ID
            )
            sampler_inputs = _spatial_sampler_inputs(spatial)
            latent_source = (
                AnimaUpscaleLatentSourceWorkflow(
                    FULL_WIDTH,
                    FULL_HEIGHT,
                    TARGET_WIDTH,
                    TARGET_HEIGHT,
                )
                if case.expect_success
                else UnsampledAnimaUpscaleLatentSourceWorkflow(
                    FULL_WIDTH,
                    FULL_HEIGHT,
                    TARGET_WIDTH,
                    TARGET_HEIGHT,
                )
            )
            denoise = REFINEMENT_DENOISE
        built = AnimaAttentionCouplingWorkflowBuilder(
            public_node_id=node_id,
            artifact_phase="p9.7",
            conditioning=ScheduledRegionalPromptConditioningWorkflow(
                global_loras=(),
                regional_loras=case.regional_loras,
            ),
            steps=STEPS,
            sampler_inputs=sampler_inputs,
            model_modifier_workflow=RegionalPatchModifierWorkflow(case.modifier),
            model_observer_workflow=RegionalPatchModelEvidenceWorkflow(
                f"{run_id}:{case.case_id}:modifier"
            ),
            latent_source=latent_source,
            denoise=denoise,
        ).build(case, run_id=run_id, mask_names=mask_names)
        return _from_anima(built, node_id=node_id)

    @staticmethod
    def _build_sdxl(
        case: RegionalPatchInteropCase,
        *,
        run_id: str,
        mask_names: tuple[str, ...],
        checkpoint_name: str,
    ) -> BuiltRegionalPatchInteropWorkflow:
        """Build the focused SDXL NegPiP rejection graph."""

        graph = AnimaWorkflowGraph()
        loader = graph.add("CheckpointLoaderSimple", ckpt_name=checkpoint_name)
        negpip = graph.add("CLIPNegPip", model=[loader, 0], clip=[loader, 1])
        snapshot_run_id = f"{run_id}:{case.case_id}:modifier"
        snapshot = graph.add(
            "SimpleSyrupBenchmark.SnapshotModelModifier",
            model=[negpip, 0],
            run_id=snapshot_run_id,
        )
        positive = _conditioning_batch(
            graph,
            clip=[negpip, 1],
            prompts=SDXL_POSITIVE_PROMPTS,
        )
        negative = _conditioning_batch(
            graph,
            clip=[negpip, 1],
            prompts=SDXL_NEGATIVE_PROMPTS,
        )
        masks = graph.add(
            "SimpleSyrup.LoadMaskBatch",
            image={"__value__": list(mask_names)},
            channel="red",
        )
        latent = graph.add(
            "EmptyLatentImage",
            width=FULL_WIDTH,
            height=FULL_HEIGHT,
            batch_size=1,
        )
        metrics_run_id = f"{run_id}:{case.case_id}"
        instrumented = graph.add(
            "SimpleSyrupBenchmark.InstrumentModel",
            model=[snapshot, 0],
            run_id=metrics_run_id,
        )
        diagnostics_run_id = f"{metrics_run_id}:regional-diagnostics"
        captured = graph.add(
            "SimpleSyrupBenchmark.CaptureRegionalDiagnostics",
            model=[instrumented, 0],
            run_id=diagnostics_run_id,
        )
        sampled = graph.add(
            FULL_NODE_ID,
            model=[captured, 0],
            seed=7_429_113_057,
            steps=STEPS,
            cfg=case.cfg,
            sampler_name="dpmpp_2m_sde",
            scheduler="karras",
            positive=positive,
            negative=negative,
            region_masks=[masks, 0],
            regional_prompt_weight=1.0,
            region_mask_feather=case.feather,
            latent_image=[latent, 0],
            denoise=1.0,
        )
        metrics = graph.add(
            "SimpleSyrupBenchmark.ReadMetrics",
            latent=[sampled, 0],
            run_id=metrics_run_id,
        )
        diagnostics = graph.add(
            "SimpleSyrupBenchmark.ReadRegionalDiagnostics",
            latent=[metrics, 0],
            run_id=diagnostics_run_id,
        )
        decoded = graph.add("VAEDecode", samples=[diagnostics, 0], vae=[loader, 2])
        saved = graph.add(
            "SaveImage",
            images=[decoded, 0],
            filename_prefix=f"simple_syrup_p9.7/{run_id}/{case.case_id}",
        )
        return BuiltRegionalPatchInteropWorkflow(
            graph.prompt,
            sampled,
            snapshot,
            saved,
            metrics,
            diagnostics,
            metrics_run_id,
            diagnostics_run_id,
        )


def _from_anima(
    built: BuiltAnimaAttentionCouplingWorkflow,
    *,
    node_id: str,
) -> BuiltRegionalPatchInteropWorkflow:
    """Narrow shared Anima output identities into the P9.7 contract."""

    sampler_ids = tuple(
        graph_node_id
        for graph_node_id, node in built.prompt.items()
        if node.get("class_type") == node_id
    )
    if len(sampler_ids) != 1:
        raise ValueError("P9.7 workflow must contain one target public sampler.")
    if built.modifier_snapshot_node_id is None:
        raise ValueError("P9.7 workflow is missing MODEL modifier evidence.")
    return BuiltRegionalPatchInteropWorkflow(
        built.prompt,
        sampler_ids[0],
        built.modifier_snapshot_node_id,
        built.save_node_id,
        built.metrics_node_id,
        built.diagnostics_node_id,
        built.metrics_run_id,
        built.diagnostics_run_id,
        built.source_save_node_id,
    )


def _spatial_sampler_inputs(
    mode: PatchInteropSpatialMode,
) -> dict[str, object]:
    """Return established tiled or Contextual controls without new policy."""

    if mode is PatchInteropSpatialMode.TILED:
        return {
            "diffusion_mode": "multidiffusion",
            "latent_tile_width": TILE_WIDTH,
            "latent_tile_height": TILE_HEIGHT,
            "latent_tile_overlap": TILE_OVERLAP,
            "latent_tile_batch_size": 4,
        }
    if mode is PatchInteropSpatialMode.CONTEXTUAL:
        return {
            "diffusion_mode": "multidiffusion",
            "latent_context_size": CONTEXT_SIZE,
            "latent_context_overlap": CONTEXT_OVERLAP,
            "latent_context_batch_size": CONTEXT_BATCH_SIZE,
            "global_weight": GLOBAL_WEIGHT,
            "global_steps": 3,
            "global_decay": GLOBAL_DECAY,
        }
    raise ValueError("P9.7 spatial inputs require tiled or Contextual mode.")


def _conditioning_batch(
    graph: AnimaWorkflowGraph,
    *,
    clip: NodeReference,
    prompts: tuple[str, ...],
) -> NodeReference:
    """Encode and pack one global-first SDXL conditioning batch."""

    encoded = tuple(
        graph.add("CLIPTextEncode", clip=clip, text=prompt) for prompt in prompts
    )
    current = graph.add(
        "SimpleSyrup.ConditioningBatchStart",
        conditioning=[encoded[0], 0],
    )
    for node in encoded[1:]:
        current = graph.add(
            "SimpleSyrup.ConditioningBatchAppend",
            batch=[current, 0],
            conditioning=[node, 0],
        )
    return [current, 0]
