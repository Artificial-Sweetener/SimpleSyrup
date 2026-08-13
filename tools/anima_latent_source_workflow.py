# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build empty and coherent-upscale latent sources for Anima workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tools.anima_workflow_graph import AnimaWorkflowGraph, NodeReference


@dataclass(frozen=True, slots=True)
class AnimaLatentSourceResult:
    """Return the refinement latent and optional reviewable source image."""

    latent: NodeReference
    source_image: NodeReference | None = None


class AnimaLatentSourceWorkflow(Protocol):
    """Build one latent source without owning the consuming sampler graph."""

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
        """Add source nodes and return the latent plus reviewable source image."""


@dataclass(frozen=True, slots=True)
class EmptyAnimaLatentSourceWorkflow:
    """Build one ordinary empty Anima image latent."""

    width: int
    height: int

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
        """Add the host empty-latent node; unrelated inputs remain explicit."""

        del (
            model,
            positive,
            negative,
            region_masks,
            vae,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            region_mask_feather,
        )
        latent = graph.add(
            "EmptyCosmosLatentVideo",
            width=self.width,
            height=self.height,
            length=1,
            batch_size=1,
        )
        return AnimaLatentSourceResult([latent, 0])


@dataclass(frozen=True, slots=True)
class AnimaUpscaleLatentSourceWorkflow:
    """Generate a coherent source, upscale it, and encode a refinement latent."""

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
        """Add full-context generation and resize to the requested target canvas."""

        source_latent = graph.add(
            "EmptyCosmosLatentVideo",
            width=self.source_width,
            height=self.source_height,
            length=1,
            batch_size=1,
        )
        source_sample = graph.add(
            "SimpleSyrup.KSamplerAttentionCoupling",
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            region_masks=region_masks,
            regional_prompt_weight=1.0,
            region_mask_feather=region_mask_feather,
            latent_image=[source_latent, 0],
            denoise=1.0,
        )
        decoded = graph.add("VAEDecode", samples=[source_sample, 0], vae=vae)
        upscaled = graph.add(
            "ImageScale",
            image=[decoded, 0],
            upscale_method="lanczos",
            width=self.target_width,
            height=self.target_height,
            crop="disabled",
        )
        encoded = graph.add("VAEEncode", pixels=[upscaled, 0], vae=vae)
        return AnimaLatentSourceResult([encoded, 0], [decoded, 0])
