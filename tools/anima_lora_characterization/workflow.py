# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build ordinary global Comfy workflows for a selected Anima LoRA."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tools.attention_coupling_benchmark.manifest_types import SamplingSettings
from tools.comfy_api import JsonObject

from .matrix import AdapterDefinition, LoraRun

Workflow = dict[str, JsonObject]


@dataclass(frozen=True)
class BuiltLoraWorkflow:
    """Return one prompt graph and its result-producing node identities."""

    prompt: Workflow
    metrics_node_id: str
    save_node_id: str


class LoraWorkflowBuilder:
    """Translate one fixed profile into ordinary Comfy loader and hook nodes."""

    def __init__(self, lora_name: str) -> None:
        """Retain the externally selected ComfyUI-relative adapter name."""

        if not lora_name or Path(lora_name).is_absolute():
            raise ValueError("LoRA name must be non-empty and ComfyUI-relative.")
        self._lora_name = lora_name

    def build(
        self,
        run: LoraRun,
        sampling: SamplingSettings,
        positive_prompt: str,
    ) -> BuiltLoraWorkflow:
        """Build a complete loader-to-image API graph."""

        graph = _Graph()
        loader = graph.add(
            "SimpleSyrup.SimpleLoadAnima",
            diffusion_model="Anima\\diffusion-model.safetensors",
            quantization="Original",
            diffusion_weight_dtype="default",
            text_encoder="qwen\\qwen_3_06b_base.safetensors",
            text_encoder_device="default",
            vae="qwen\\qwen_image_vae.safetensors",
        )
        model_link: object = [loader, 0]
        if run.profile.mode == "static":
            for adapter in run.profile.adapters:
                node_id = graph.add(
                    "LoraLoaderModelOnly",
                    model=model_link,
                    lora_name=self._lora_name,
                    strength_model=adapter.strength,
                )
                model_link = [node_id, 0]
        positive_encode = graph.add(
            "CLIPTextEncode",
            clip=[loader, 1],
            text=positive_prompt,
        )
        negative_encode = graph.add(
            "CLIPTextEncode",
            clip=[loader, 1],
            text=sampling.negative_prompt,
        )
        positive: object = [positive_encode, 0]
        negative: object = [negative_encode, 0]
        if run.profile.mode == "scheduled":
            hooks = [
                self._scheduled_hook(graph, adapter) for adapter in run.profile.adapters
            ]
            combined = self._combine_hooks(graph, hooks)
            attached = graph.add(
                "PairConditioningSetProperties",
                positive_NEW=positive,
                negative_NEW=negative,
                strength=1.0,
                set_cond_area="default",
                hooks=[combined, 0],
            )
            positive = [attached, 0]
            negative = [attached, 1]
        instrument = graph.add(
            "SimpleSyrupBenchmark.InstrumentLoraModel",
            model=model_link,
            run_id=run.artifact_id,
            adapter_identities_json=json.dumps(
                [adapter.identity for adapter in run.profile.adapters]
            ),
            capture_outputs=run.capture_outputs,
        )
        latent = graph.add(
            "EmptyCosmosLatentVideo",
            width=sampling.width,
            height=sampling.height,
            length=1,
            batch_size=1,
        )
        sampled = graph.add(
            "KSampler",
            model=[instrument, 0],
            seed=run.seed,
            steps=sampling.steps,
            cfg=sampling.cfg,
            sampler_name=sampling.sampler,
            scheduler=sampling.scheduler,
            positive=positive,
            negative=negative,
            latent_image=[latent, 0],
            denoise=sampling.denoise,
        )
        metrics = graph.add(
            "SimpleSyrupBenchmark.ReadLoraMetrics",
            latent=[sampled, 0],
            run_id=run.artifact_id,
        )
        decoded = graph.add("VAEDecode", samples=[metrics, 0], vae=[loader, 2])
        saved = graph.add(
            "SaveImage",
            images=[decoded, 0],
            filename_prefix=f"simple_syrup_benchmark/primary_adapter-global/{run.artifact_id}",
        )
        return BuiltLoraWorkflow(graph.prompt, metrics, saved)

    def _scheduled_hook(self, graph: _Graph, adapter: AdapterDefinition) -> str:
        """Create one independently scheduled model-only WeightHook."""

        hook = graph.add(
            "CreateHookLoraModelOnly",
            lora_name=self._lora_name,
            strength_model=adapter.strength,
        )
        previous: object | None = None
        for start_percent, strength_mult in adapter.schedule:
            inputs: dict[str, object] = {
                "strength_mult": strength_mult,
                "start_percent": start_percent,
            }
            if previous is not None:
                inputs["prev_hook_kf"] = previous
            keyframe = graph.add("CreateHookKeyframe", **inputs)
            previous = [keyframe, 0]
        if previous is None:
            raise ValueError(
                f"Scheduled adapter has no keyframes: {adapter.identity!r}."
            )
        return graph.add("SetHookKeyframes", hooks=[hook, 0], hook_kf=previous)

    @staticmethod
    def _combine_hooks(graph: _Graph, hook_ids: list[str]) -> str:
        """Combine only the supported one, two, or four ordered hook stacks."""

        if len(hook_ids) == 1:
            return hook_ids[0]
        if len(hook_ids) == 2:
            return graph.add(
                "CombineHooks2", hooks_A=[hook_ids[0], 0], hooks_B=[hook_ids[1], 0]
            )
        if len(hook_ids) == 4:
            return graph.add(
                "CombineHooks4",
                **{
                    f"hooks_{letter}": [hook_id, 0]
                    for letter, hook_id in zip("ABCD", hook_ids, strict=True)
                },
            )
        raise ValueError(
            "Scheduled LoRA profiles must contain one, two, or four hooks."
        )


class _Graph:
    """Own sequential node identities within one generated API graph."""

    def __init__(self) -> None:
        """Initialize an empty deterministic graph."""

        self.prompt: Workflow = {}

    def add(self, class_type: str, **inputs: object) -> str:
        """Append one node and return its stable numeric identity."""

        node_id = str(len(self.prompt) + 1)
        self.prompt[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id
