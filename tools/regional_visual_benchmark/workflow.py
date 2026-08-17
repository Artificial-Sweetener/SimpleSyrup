# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build public-node workflows for corrected P10.3 visual positions."""

from __future__ import annotations

from dataclasses import dataclass

from tools.anima_workflow_graph import AnimaWorkflowGraph, NodeReference
from tools.attention_coupling_benchmark.manifest_types import (
    BenchmarkCase,
    BenchmarkManifest,
)
from tools.comfy_api import JsonObject
from tools.comfy_integration.anima_fixture_selections import (
    ANIMA_DIFFUSION_SELECTION,
    ANIMA_TEXT_ENCODER_SELECTION,
    ANIMA_VAE_SELECTION,
)

from .matrix import (
    CONTEXTUAL_GLOBAL_STEPS,
    REFINEMENT_DENOISE,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    SPATIAL_BATCH_SIZE,
    SPATIAL_OVERLAP,
    SPATIAL_SIZE,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    RegionalStrategy,
    SourcePosition,
    SpatialProfile,
    VisualPosition,
)


@dataclass(frozen=True, slots=True)
class BuiltVisualWorkflow:
    """Expose one graph and its evidence-producing node identities."""

    prompt: dict[str, JsonObject]
    metrics_node_id: str
    save_node_id: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every exact node contract named by this graph."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


class VisualBenchmarkWorkflowBuilder:
    """Translate typed visual positions into deterministic Comfy graphs."""

    def build_source(
        self,
        manifest: BenchmarkManifest,
        source: SourcePosition,
        *,
        run_id: str,
    ) -> BuiltVisualWorkflow:
        """Build one strategy-neutral global-prompt source workflow."""

        case = self._case(manifest, source.case_id)
        graph = AnimaWorkflowGraph()
        loader = self._loader(graph)
        positive = graph.add(
            "CLIPTextEncode", clip=[loader, 1], text=self._source_prompt(case)
        )
        negative = graph.add(
            "CLIPTextEncode",
            clip=[loader, 1],
            text=manifest.sampling.negative_prompt,
        )
        latent = graph.add(
            "EmptyCosmosLatentVideo",
            width=SOURCE_WIDTH,
            height=SOURCE_HEIGHT,
            length=1,
            batch_size=1,
        )
        metrics_run_id = f"{run_id}:{source.source_id}"
        instrumented = graph.add(
            "SimpleSyrupBenchmark.InstrumentModel",
            model=[loader, 0],
            run_id=metrics_run_id,
        )
        sampled = graph.add(
            "KSampler",
            model=[instrumented, 0],
            seed=source.seed,
            steps=manifest.sampling.steps,
            cfg=manifest.sampling.cfg,
            sampler_name=manifest.sampling.sampler,
            scheduler=manifest.sampling.scheduler,
            positive=[positive, 0],
            negative=[negative, 0],
            latent_image=[latent, 0],
            denoise=1.0,
        )
        return self._outputs(
            graph,
            latent=[sampled, 0],
            vae=[loader, 2],
            metrics_run_id=metrics_run_id,
            filename_prefix=f"simple_syrup_p10.3/{run_id}/sources/{source.source_id}",
        )

    def build_visual(
        self,
        manifest: BenchmarkManifest,
        position: VisualPosition,
        *,
        run_id: str,
        mask_names: tuple[str, ...],
        source_name: str | None,
    ) -> BuiltVisualWorkflow:
        """Build one full or corrected refinement strategy workflow."""

        case = self._case(manifest, position.case_id)
        if len(mask_names) != len(case.masks):
            raise ValueError("P10.3 materialized mask count does not match the case.")
        if position.is_refinement != (source_name is not None):
            raise ValueError("P10.3 source publication does not match the profile.")
        graph = AnimaWorkflowGraph()
        loader = self._loader(graph)
        encoded = graph.add(
            "SimpleSyrup.EncodePromptBatch",
            clip=[loader, 1],
            positive_prompt="[SEP]".join((case.global_prompt, *case.regional_prompts)),
            negative_prompt=manifest.sampling.negative_prompt,
            separator="[SEP]",
        )
        masks = self._masks(graph, manifest, mask_names)
        latent = self._latent(
            graph,
            position,
            source_name=source_name,
            vae=[loader, 2],
        )
        metrics_run_id = f"{run_id}:{position.artifact_id}"
        model = graph.add(
            "SimpleSyrupBenchmark.InstrumentModel",
            model=[loader, 0],
            run_id=metrics_run_id,
        )
        sampled = graph.add(
            self._sampler_node_id(position),
            **self._sampler_inputs(
                manifest,
                case,
                position,
                model=[model, 0],
                positive=[encoded, 0],
                negative=[encoded, 1],
                masks=[masks, 0],
                latent=latent,
            ),
        )
        return self._outputs(
            graph,
            latent=[sampled, 0],
            vae=[loader, 2],
            metrics_run_id=metrics_run_id,
            filename_prefix=(
                f"simple_syrup_p10.3/{run_id}/outputs/{position.artifact_id}"
            ),
        )

    @staticmethod
    def _loader(graph: AnimaWorkflowGraph) -> str:
        """Add the pinned Anima stack loader."""

        return graph.add(
            "SimpleSyrup.SimpleLoadAnima",
            diffusion_model=ANIMA_DIFFUSION_SELECTION,
            quantization="Original",
            diffusion_weight_dtype="default",
            text_encoder=ANIMA_TEXT_ENCODER_SELECTION,
            text_encoder_device="default",
            vae=ANIMA_VAE_SELECTION,
        )

    @staticmethod
    def _source_prompt(case: BenchmarkCase) -> str:
        """Join exact case texts for strategy-neutral source generation."""

        return ", ".join((case.global_prompt, *case.regional_prompts))

    @staticmethod
    def _masks(
        graph: AnimaWorkflowGraph,
        manifest: BenchmarkManifest,
        mask_names: tuple[str, ...],
    ) -> str:
        """Load authored masks or add the global-only technical zero mask."""

        if mask_names:
            return graph.add(
                "SimpleSyrup.LoadMaskBatch",
                image={"__value__": list(mask_names)},
                channel="red",
            )
        return graph.add(
            "SolidMask",
            value=0.0,
            width=manifest.sampling.width,
            height=manifest.sampling.height,
        )

    @staticmethod
    def _latent(
        graph: AnimaWorkflowGraph,
        position: VisualPosition,
        *,
        source_name: str | None,
        vae: NodeReference,
    ) -> NodeReference:
        """Build an empty full latent or exact 1.5x source refinement latent."""

        if not position.is_refinement:
            empty = graph.add(
                "EmptyCosmosLatentVideo",
                width=SOURCE_WIDTH,
                height=SOURCE_HEIGHT,
                length=1,
                batch_size=1,
            )
            return [empty, 0]
        if source_name is None:
            raise ValueError("P10.3 refinement requires a published source name.")
        loaded = graph.add("LoadImage", image=source_name)
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
    def _sampler_node_id(position: VisualPosition) -> str:
        """Return the public sampler ID for the exact strategy/profile pair."""

        attention = position.strategy is RegionalStrategy.ATTENTION_COUPLING
        if position.spatial_profile is SpatialProfile.FULL:
            return (
                "SimpleSyrup.KSamplerAttentionCoupling"
                if attention
                else "SimpleSyrup.KSamplerPromptByRegion"
            )
        if position.spatial_profile in {
            SpatialProfile.TILED_MULTIDIFFUSION,
            SpatialProfile.TILED_MIXTURE_OF_DIFFUSERS,
        }:
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
        manifest: BenchmarkManifest,
        case: BenchmarkCase,
        position: VisualPosition,
        *,
        model: NodeReference,
        positive: NodeReference,
        negative: NodeReference,
        masks: NodeReference,
        latent: NodeReference,
    ) -> dict[str, object]:
        """Return exact public inputs for one corrected matrix position."""

        inputs: dict[str, object] = {
            "model": model,
            "seed": position.seed,
            "steps": manifest.sampling.steps,
            "cfg": manifest.sampling.cfg,
            "sampler_name": manifest.sampling.sampler,
            "scheduler": manifest.sampling.scheduler,
            "positive": positive,
            "negative": negative,
            "region_masks": masks,
            "regional_prompt_weight": case.regional_prompt_weight,
            "region_mask_feather": case.region_mask_feather,
            "latent_image": latent,
            "denoise": REFINEMENT_DENOISE if position.is_refinement else 1.0,
        }
        profile = position.spatial_profile
        if profile in {
            SpatialProfile.TILED_MULTIDIFFUSION,
            SpatialProfile.TILED_MIXTURE_OF_DIFFUSERS,
        }:
            inputs.update(
                {
                    "diffusion_mode": profile.diffusion_mode,
                    "latent_tile_width": SPATIAL_SIZE,
                    "latent_tile_height": SPATIAL_SIZE,
                    "latent_tile_overlap": SPATIAL_OVERLAP,
                    "latent_tile_batch_size": SPATIAL_BATCH_SIZE,
                }
            )
        elif profile is not SpatialProfile.FULL:
            inputs.update(
                {
                    "diffusion_mode": profile.diffusion_mode,
                    "latent_context_size": SPATIAL_SIZE,
                    "latent_context_overlap": SPATIAL_OVERLAP,
                    "latent_context_batch_size": SPATIAL_BATCH_SIZE,
                    "global_weight": 1.0,
                    "global_steps": CONTEXTUAL_GLOBAL_STEPS,
                    "global_decay": 0.5,
                }
            )
        return inputs

    @staticmethod
    def _outputs(
        graph: AnimaWorkflowGraph,
        *,
        latent: NodeReference,
        vae: NodeReference,
        metrics_run_id: str,
        filename_prefix: str,
    ) -> BuiltVisualWorkflow:
        """Append metrics, decode, and save nodes in one stable order."""

        metrics = graph.add(
            "SimpleSyrupBenchmark.ReadMetrics",
            latent=latent,
            run_id=metrics_run_id,
        )
        decoded = graph.add("VAEDecode", samples=[metrics, 0], vae=vae)
        saved = graph.add(
            "SaveImage", images=[decoded, 0], filename_prefix=filename_prefix
        )
        return BuiltVisualWorkflow(graph.prompt, metrics, saved)

    @staticmethod
    def _case(manifest: BenchmarkManifest, case_id: str) -> BenchmarkCase:
        """Resolve one frozen case by stable identity."""

        return next(case for case in manifest.cases if case.case_id == case_id)
