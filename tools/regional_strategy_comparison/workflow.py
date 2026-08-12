"""Build P10.2 public-node strategy comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass

from tools.anima_attention_coupling_conditioning import (
    ScheduledRegionalPromptConditioningWorkflow,
)
from tools.anima_workflow_graph import AnimaWorkflowGraph, NodeReference
from tools.comfy_api import JsonObject

from .matrix import (
    CFG,
    CONTEXTUAL_GLOBAL_STEPS,
    REFINEMENT_DENOISE,
    SAMPLER,
    SCHEDULER,
    SEED,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    SPATIAL_BATCH_SIZE,
    SPATIAL_OVERLAP,
    SPATIAL_SIZE,
    STEPS,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    StrategyComparisonCase,
)


@dataclass(frozen=True, slots=True)
class BuiltStrategyComparisonWorkflow:
    """Expose one graph and its evidence-producing output identities."""

    prompt: dict[str, JsonObject]
    save_node_id: str
    metrics_node_id: str
    diagnostics_node_id: str | None
    metrics_run_id: str
    diagnostics_run_id: str | None

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return all exact node IDs named by the graph."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


class StrategyComparisonWorkflowBuilder:
    """Translate one matrix case into an instrumented public-node graph."""

    def build(
        self,
        case: StrategyComparisonCase,
        *,
        run_id: str,
        mask_names: tuple[str, ...],
        shared_source_name: str,
    ) -> BuiltStrategyComparisonWorkflow:
        """Build one deterministic loader-to-image comparison workflow."""

        if len(mask_names) != 2:
            raise ValueError("P10.2 requires two ordered mask files.")
        graph = AnimaWorkflowGraph()
        loader = graph.add(
            "SimpleSyrup.SimpleLoadAnima",
            diffusion_model="Anima\\diffusion-model.safetensors",
            quantization="Original",
            diffusion_weight_dtype="default",
            text_encoder="qwen\\qwen_3_06b_base.safetensors",
            text_encoder_device="default",
            vae="qwen\\qwen_image_vae.safetensors",
        )
        conditioning = ScheduledRegionalPromptConditioningWorkflow(
            global_loras=(),
            regional_loras=case.regional_loras,
        ).add(graph, model=[loader, 0], clip=[loader, 1])
        masks = graph.add(
            "SimpleSyrup.LoadMaskBatch",
            image={"__value__": list(mask_names)},
            channel="red",
        )
        latent = self._latent(graph, case, shared_source_name, vae=[loader, 2])
        metrics_run_id = f"{run_id}:{case.case_id}"
        model = graph.add(
            "SimpleSyrupBenchmark.InstrumentModel",
            model=conditioning.model,
            run_id=metrics_run_id,
        )
        diagnostics_node_id: str | None = None
        diagnostics_run_id: str | None = None
        if case.strategy == "attention_coupling":
            diagnostics_run_id = f"{metrics_run_id}:regional-diagnostics"
            model = graph.add(
                "SimpleSyrupBenchmark.CaptureRegionalDiagnostics",
                model=[model, 0],
                run_id=diagnostics_run_id,
            )
        sampled = graph.add(
            self._sampler_node_id(case),
            **self._sampler_inputs(
                case,
                model=[model, 0],
                positive=conditioning.positive,
                negative=conditioning.negative,
                latent=latent,
                masks=masks,
            ),
        )
        metrics_node_id = graph.add(
            "SimpleSyrupBenchmark.ReadMetrics",
            latent=[sampled, 0],
            run_id=metrics_run_id,
        )
        latent_output: NodeReference = [metrics_node_id, 0]
        if diagnostics_run_id is not None:
            diagnostics_node_id = graph.add(
                "SimpleSyrupBenchmark.ReadRegionalDiagnostics",
                latent=latent_output,
                run_id=diagnostics_run_id,
            )
            latent_output = [diagnostics_node_id, 0]
        decoded = graph.add("VAEDecode", samples=latent_output, vae=[loader, 2])
        save_node_id = graph.add(
            "SaveImage",
            images=[decoded, 0],
            filename_prefix=f"simple_syrup_p10.2/{run_id}/{case.case_id}",
        )
        return BuiltStrategyComparisonWorkflow(
            graph.prompt,
            save_node_id,
            metrics_node_id,
            diagnostics_node_id,
            metrics_run_id,
            diagnostics_run_id,
        )

    @staticmethod
    def _latent(
        graph: AnimaWorkflowGraph,
        case: StrategyComparisonCase,
        shared_source_name: str,
        *,
        vae: NodeReference,
    ) -> NodeReference:
        """Build the full empty latent or shared 1.5x refinement latent."""

        if not case.is_refinement:
            empty = graph.add(
                "EmptyCosmosLatentVideo",
                width=SOURCE_WIDTH,
                height=SOURCE_HEIGHT,
                length=1,
                batch_size=1,
            )
            return [empty, 0]
        loaded = graph.add("LoadImage", image=shared_source_name)
        scaled = graph.add(
            "ImageScale",
            image=[loaded, 0],
            upscale_method="lanczos",
            width=TARGET_WIDTH,
            height=TARGET_HEIGHT,
            crop="disabled",
        )
        encoded = graph.add("VAEEncode", pixels=[scaled, 0], vae=vae)
        return [encoded, 0]

    @staticmethod
    def _sampler_node_id(case: StrategyComparisonCase) -> str:
        """Return the exact public node ID for one strategy and spatial mode."""

        attention = case.strategy == "attention_coupling"
        if case.spatial_profile == "full":
            return (
                "SimpleSyrup.KSamplerAttentionCoupling"
                if attention
                else "SimpleSyrup.KSamplerPromptByRegion"
            )
        if case.spatial_profile.startswith("tiled_"):
            return (
                "SimpleSyrup.KSamplerAttentionCouplingTiled"
                if attention
                else "SimpleSyrup.KSamplerPromptByTiledRegion"
            )
        return (
            "SimpleSyrup.KSamplerAttentionCouplingContextual"
            if attention
            else "SimpleSyrup.KSamplerContextualDiffusion"
        )

    @staticmethod
    def _sampler_inputs(
        case: StrategyComparisonCase,
        *,
        model: NodeReference,
        positive: NodeReference,
        negative: NodeReference,
        latent: NodeReference,
        masks: str,
    ) -> dict[str, object]:
        """Return the exact public inputs for the selected spatial profile."""

        inputs: dict[str, object] = {
            "model": model,
            "seed": SEED,
            "steps": STEPS,
            "cfg": CFG,
            "sampler_name": SAMPLER,
            "scheduler": SCHEDULER,
            "positive": positive,
            "negative": negative,
            "region_masks": [masks, 0],
            "regional_prompt_weight": 1.0,
            "region_mask_feather": 0,
            "latent_image": latent,
            "denoise": REFINEMENT_DENOISE if case.is_refinement else 1.0,
        }
        if case.spatial_profile.startswith("tiled_"):
            inputs.update(
                {
                    "diffusion_mode": case.diffusion_mode,
                    "latent_tile_width": SPATIAL_SIZE,
                    "latent_tile_height": SPATIAL_SIZE,
                    "latent_tile_overlap": SPATIAL_OVERLAP,
                    "latent_tile_batch_size": SPATIAL_BATCH_SIZE,
                }
            )
        elif case.spatial_profile.startswith("contextual_"):
            inputs.update(
                {
                    "diffusion_mode": case.diffusion_mode,
                    "latent_context_size": SPATIAL_SIZE,
                    "latent_context_overlap": SPATIAL_OVERLAP,
                    "latent_context_batch_size": SPATIAL_BATCH_SIZE,
                    "global_weight": 1.0,
                    "global_steps": CONTEXTUAL_GLOBAL_STEPS,
                    "global_decay": 0.5,
                }
            )
        return inputs
