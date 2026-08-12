"""Build a no-denoiser 1024-to-1536 latent for rejection graphs."""

from __future__ import annotations

from dataclasses import dataclass

from tools.anima_latent_source_workflow import AnimaLatentSourceResult
from tools.anima_workflow_graph import AnimaWorkflowGraph, NodeReference


@dataclass(frozen=True, slots=True)
class UnsampledAnimaUpscaleLatentSourceWorkflow:
    """Decode, resize, and encode an empty source without a model invocation."""

    source_width: int
    source_height: int
    target_width: int
    target_height: int

    def add(
        self,
        graph: AnimaWorkflowGraph,
        *,
        model: NodeReference,
        positive: NodeReference,
        negative: NodeReference,
        region_masks: NodeReference,
        vae: NodeReference,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        region_mask_feather: int,
    ) -> AnimaLatentSourceResult:
        """Return a target latent while explicitly discarding sampler inputs."""

        del (
            model,
            positive,
            negative,
            region_masks,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            region_mask_feather,
        )
        source = graph.add(
            "EmptyCosmosLatentVideo",
            width=self.source_width,
            height=self.source_height,
            length=1,
            batch_size=1,
        )
        decoded = graph.add("VAEDecode", samples=[source, 0], vae=vae)
        upscaled = graph.add(
            "ImageScale",
            image=[decoded, 0],
            upscale_method="lanczos",
            width=self.target_width,
            height=self.target_height,
            crop="disabled",
        )
        encoded = graph.add("VAEEncode", pixels=[upscaled, 0], vae=vae)
        return AnimaLatentSourceResult([encoded, 0])
