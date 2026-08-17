# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build one instrumented public SDXL Attention Coupling sampler branch."""

from __future__ import annotations

from dataclasses import dataclass

from .graph import NodeReference, SdxlWorkflowGraph
from .sampling_controls import SDXL_VISUAL_SAMPLING


@dataclass(frozen=True, slots=True)
class SdxlWorkflowOutputs:
    """Retain output-node identities for one labeled sampler mode."""

    save_node_id: str
    metrics_node_id: str
    diagnostics_node_id: str


@dataclass(frozen=True, slots=True)
class SdxlSamplerBranchResult:
    """Retain one sampled latent and its already-created decoded image."""

    latent: NodeReference
    decoded: NodeReference
    outputs: SdxlWorkflowOutputs


def add_sampler_branch(
    graph: SdxlWorkflowGraph,
    *,
    mode_id: str,
    node_id: str,
    model: NodeReference,
    positive: NodeReference,
    negative: NodeReference,
    masks: NodeReference,
    latent: NodeReference,
    vae: NodeReference,
    run_id: str,
    seed: int,
    steps: int,
    denoise: float,
    regional_prompt_weight: float,
    region_mask_feather: int,
    sampler_inputs: dict[str, object],
    filename_prefix: str | None = None,
) -> SdxlSamplerBranchResult:
    """Add one sampler, diagnostics, decode, and save chain."""

    metrics_run_id = f"{run_id}:{mode_id}:metrics"
    diagnostics_run_id = f"{run_id}:{mode_id}:diagnostics"
    instrumented = graph.add(
        "SimpleSyrupBenchmark.InstrumentModel",
        model=model,
        run_id=metrics_run_id,
    )
    captured = graph.add(
        "SimpleSyrupBenchmark.CaptureRegionalDiagnostics",
        model=[instrumented, 0],
        run_id=diagnostics_run_id,
    )
    sampled = graph.add(
        node_id,
        model=[captured, 0],
        seed=seed,
        steps=steps,
        cfg=SDXL_VISUAL_SAMPLING.cfg,
        sampler_name=SDXL_VISUAL_SAMPLING.sampler,
        scheduler=SDXL_VISUAL_SAMPLING.scheduler,
        positive=positive,
        negative=negative,
        region_masks=masks,
        regional_prompt_weight=regional_prompt_weight,
        region_mask_feather=region_mask_feather,
        latent_image=latent,
        denoise=denoise,
        **sampler_inputs,
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
    decoded = graph.add("VAEDecode", samples=[diagnostics, 0], vae=vae)
    saved = graph.add(
        "SaveImage",
        images=[decoded, 0],
        filename_prefix=filename_prefix or f"simple_syrup_p8_5_sdxl/{run_id}/{mode_id}",
    )
    outputs = SdxlWorkflowOutputs(saved, metrics, diagnostics)
    return SdxlSamplerBranchResult([diagnostics, 0], [decoded, 0], outputs)
