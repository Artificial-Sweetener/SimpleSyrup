# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build one ordinary native-Comfy SDXL global-LoRA reference graph."""

from __future__ import annotations

from dataclasses import dataclass

from tools.comfy_api import JsonObject
from tools.sdxl_attention_couple_parity.cases import SdxlAttentionCoupleParityCase
from tools.sdxl_attention_coupling_integration.graph import (
    NodeReference,
    SdxlWorkflowGraph,
)
from tools.sdxl_attention_coupling_integration.matrix import (
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)
from tools.sdxl_attention_coupling_integration.sampling_controls import (
    SDXL_VISUAL_SAMPLING,
)

from .cases import NativeGlobalLoraCase

GLOBAL_LORA_STRENGTH = 0.65


@dataclass(frozen=True, slots=True)
class BuiltNativeGlobalLoraWorkflow:
    """Expose one native graph and its terminal image node."""

    case: NativeGlobalLoraCase
    prompt: dict[str, JsonObject]
    save_node_id: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every live Comfy node required by this graph."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


def build_native_global_lora_workflow(
    *,
    case: NativeGlobalLoraCase,
    run_id: str,
    checkpoint_name: str,
    lora_name: str,
    prompt_case: SdxlAttentionCoupleParityCase,
    style_prompt_g: str,
    style_prompt_l: str,
) -> BuiltNativeGlobalLoraWorkflow:
    """Build one global-prompt graph with no regional custom-node execution."""

    if not run_id.strip() or not checkpoint_name.strip() or not lora_name.strip():
        raise ValueError("Native global-LoRA graph selections must be non-empty.")
    graph = SdxlWorkflowGraph()
    loader = graph.add("CheckpointLoaderSimple", ckpt_name=checkpoint_name)
    model: NodeReference = [loader, 0]
    clip: NodeReference = [loader, 1]
    if case.applies_lora:
        loaded = graph.add(
            "LoraLoader",
            model=model,
            clip=clip,
            lora_name=lora_name,
            strength_model=GLOBAL_LORA_STRENGTH,
            strength_clip=GLOBAL_LORA_STRENGTH,
        )
        model = [loaded, 0]
        clip = [loaded, 1]
    positive = _encode(
        graph,
        clip=clip,
        text_g=_append(prompt_case.base_positive_g, style_prompt_g),
        text_l=_append(prompt_case.base_positive_l, style_prompt_l),
    )
    negative = _encode(
        graph,
        clip=clip,
        text_g=prompt_case.base_negative_g,
        text_l=prompt_case.base_negative_l,
    )
    latent = graph.add(
        "EmptyLatentImage",
        width=SOURCE_WIDTH,
        height=SOURCE_HEIGHT,
        batch_size=1,
    )
    sampled = graph.add(
        "KSampler",
        model=model,
        positive=positive,
        negative=negative,
        seed=SDXL_VISUAL_SAMPLING.seed,
        steps=SDXL_VISUAL_SAMPLING.steps,
        cfg=SDXL_VISUAL_SAMPLING.cfg,
        sampler_name=SDXL_VISUAL_SAMPLING.sampler,
        scheduler=SDXL_VISUAL_SAMPLING.scheduler,
        latent_image=[latent, 0],
        denoise=1.0,
    )
    decoded = graph.add("VAEDecode", samples=[sampled, 0], vae=[loader, 2])
    saved = graph.add(
        "SaveImage",
        images=[decoded, 0],
        filename_prefix=(f"simple_syrup_native_global_lora/{run_id}/{case.value}"),
    )
    return BuiltNativeGlobalLoraWorkflow(case, graph.prompt, saved)


def _encode(
    graph: SdxlWorkflowGraph,
    *,
    clip: NodeReference,
    text_g: str,
    text_l: str,
) -> NodeReference:
    """Encode one native SDXL G/L prompt at the locked canvas geometry."""

    encoded = graph.add(
        "CLIPTextEncodeSDXL",
        clip=clip,
        width=SOURCE_WIDTH,
        height=SOURCE_HEIGHT,
        crop_w=0,
        crop_h=0,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        text_g=text_g,
        text_l=text_l,
    )
    return [encoded, 0]


def _append(prompt: str, addition: str) -> str:
    """Append one required external trigger without changing base wording."""

    if not prompt.strip() or not addition.strip():
        raise ValueError("Native global-LoRA prompts must be non-empty.")
    return f"{prompt}, {addition}"
