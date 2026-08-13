# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build Prompt Control characterization graphs from public Comfy nodes."""

from __future__ import annotations

import json
from dataclasses import dataclass

from tools.anima_workflow_graph import AnimaWorkflowGraph, NodeReference
from tools.comfy_api import JsonObject

from .cases import PromptControlCase
from .conditioning import (
    PROMPT_CONTROL_TEXT_CONDITIONING_BUILDER,
    expansion_inputs,
)

Workflow = dict[str, JsonObject]


@dataclass(frozen=True)
class BuiltPromptControlWorkflow:
    """Return one graph and its two evidence-producing node identities."""

    prompt: Workflow
    expansion_node_id: str
    snapshot_node_id: str
    runtime_node_id: str


class PromptControlWorkflowBuilder:
    """Compose public Prompt Control nodes with native Comfy sampling."""

    def build(self, case: PromptControlCase) -> BuiltPromptControlWorkflow:
        """Build one complete loader-to-runtime-evidence API graph."""

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
        expansion_spec = expansion_inputs(case)
        expansion = graph.add(
            "SimpleSyrupBenchmark.SnapshotPromptControlExpansion",
            case_id=case.case_id,
            text=expansion_spec.text,
            capture_text_expansion=expansion_spec.capture_text_expansion,
            lora_text=case.lora_text,
        )
        model: NodeReference = [loader, 0]
        clip: NodeReference = [loader, 1]
        lora_loader: str | None = None
        if case.lora_text:
            lora_loader = graph.add(
                "PCLazyLoraLoaderAdvanced",
                model=model,
                clip=clip,
                text=case.lora_text,
                apply_hooks=True,
                tags="",
                start=0.0,
                end=1.0,
                num_steps=0,
            )
            model = [lora_loader, 0]
            clip = [lora_loader, 1]
        positive = PROMPT_CONTROL_TEXT_CONDITIONING_BUILDER.add(
            graph,
            case,
            clip=clip,
        )
        negative_encode = graph.add(
            "PCTextEncodeWithRange",
            clip=clip,
            text="low quality",
            start=0.0,
            end=1.0,
        )
        identities = [adapter.identity for adapter in case.expected_adapters]
        snapshot_inputs: dict[str, object] = {
            "positive": positive,
            "negative": [negative_encode, 0],
            "case_id": case.case_id,
            "adapter_identities_json": json.dumps(identities),
        }
        if lora_loader is not None and case.expected_adapters:
            snapshot_inputs["hooks"] = [lora_loader, 2]
        snapshot = graph.add(
            "SimpleSyrupBenchmark.SnapshotPromptControl", **snapshot_inputs
        )
        instrument = graph.add(
            "SimpleSyrupBenchmark.InstrumentPromptControlModel",
            model=model,
            run_id=case.case_id,
            adapter_identities_json=json.dumps(identities),
        )
        latent = graph.add(
            "EmptyCosmosLatentVideo",
            width=512,
            height=512,
            length=1,
            batch_size=1,
        )
        sampled = graph.add(
            "KSampler",
            model=[instrument, 0],
            seed=1_029_384_756,
            steps=8,
            cfg=4.0,
            sampler_name="er_sde",
            scheduler="simple",
            positive=[snapshot, 0],
            negative=[snapshot, 1],
            latent_image=[latent, 0],
            denoise=1.0,
        )
        runtime = graph.add(
            "SimpleSyrupBenchmark.ReadPromptControlRuntime",
            latent=[sampled, 0],
            run_id=case.case_id,
        )
        return BuiltPromptControlWorkflow(graph.prompt, expansion, snapshot, runtime)
