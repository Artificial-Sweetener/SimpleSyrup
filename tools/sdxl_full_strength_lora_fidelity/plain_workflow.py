# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build one ordinary no-LoRA, no-region SDXL KSampler comparator."""

from __future__ import annotations

from dataclasses import dataclass

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.graph import (
    NodeReference,
    SdxlWorkflowGraph,
)
from tools.sdxl_attention_coupling_integration.sampling_controls import (
    SDXL_VISUAL_SAMPLING,
)


@dataclass(frozen=True, slots=True)
class BuiltPlainKSamplerWorkflow:
    """Expose the ordinary graph and exact terminal evidence nodes."""

    prompt: dict[str, JsonObject]
    save_node_id: str
    metrics_node_id: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every live Comfy node required by this comparator."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


def build_plain_ksampler_workflow(
    *,
    run_id: str,
    checkpoint_name: str,
    positive_g: str,
    positive_l: str,
    negative_g: str,
    negative_l: str,
) -> BuiltPlainKSamplerWorkflow:
    """Build a native SDXL conditioning and ordinary KSampler graph."""

    if any(
        not value.strip()
        for value in (
            run_id,
            checkpoint_name,
            positive_g,
            positive_l,
            negative_g,
            negative_l,
        )
    ):
        raise ValueError("Plain KSampler controls must be non-empty.")
    graph = SdxlWorkflowGraph()
    loader = graph.add("CheckpointLoaderSimple", ckpt_name=checkpoint_name)
    positive = _encode(graph, [loader, 1], positive_g, positive_l)
    negative = _encode(graph, [loader, 1], negative_g, negative_l)
    latent = graph.add("EmptyLatentImage", width=1024, height=1024, batch_size=1)
    metrics_run_id = f"{run_id}:plain-ksampler:metrics"
    instrumented = graph.add(
        "SimpleSyrupBenchmark.InstrumentModel",
        model=[loader, 0],
        run_id=metrics_run_id,
    )
    sampled = graph.add(
        "KSampler",
        model=[instrumented, 0],
        seed=SDXL_VISUAL_SAMPLING.seed,
        steps=SDXL_VISUAL_SAMPLING.steps,
        cfg=SDXL_VISUAL_SAMPLING.cfg,
        sampler_name=SDXL_VISUAL_SAMPLING.sampler,
        scheduler=SDXL_VISUAL_SAMPLING.scheduler,
        positive=positive,
        negative=negative,
        latent_image=[latent, 0],
        denoise=1.0,
    )
    metrics = graph.add(
        "SimpleSyrupBenchmark.ReadMetrics",
        latent=[sampled, 0],
        run_id=metrics_run_id,
    )
    decoded = graph.add("VAEDecode", samples=[metrics, 0], vae=[loader, 2])
    saved = graph.add(
        "SaveImage",
        images=[decoded, 0],
        filename_prefix=f"simple_syrup_plain_ksampler/{run_id}/pink-control",
    )
    return BuiltPlainKSamplerWorkflow(graph.prompt, saved, metrics)


def _encode(
    graph: SdxlWorkflowGraph,
    clip: NodeReference,
    text_g: str,
    text_l: str,
) -> NodeReference:
    """Encode one native SDXL G/L prompt at locked geometry."""

    node = graph.add(
        "CLIPTextEncodeSDXL",
        clip=clip,
        width=1024,
        height=1024,
        crop_w=0,
        crop_h=0,
        target_width=1536,
        target_height=1536,
        text_g=text_g,
        text_l=text_l,
    )
    return [node, 0]
