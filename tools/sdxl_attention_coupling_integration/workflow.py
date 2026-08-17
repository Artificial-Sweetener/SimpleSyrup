# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build the public full, tiled, and Contextual SDXL workflow graph."""

from __future__ import annotations

from dataclasses import dataclass

from tools.comfy_api import JsonObject

from .graph import SdxlWorkflowGraph
from .matrix import (
    MODES,
    NEGATIVE_PROMPTS,
    NEGATIVE_PROMPTS_G,
    POSITIVE_PROMPTS,
    POSITIVE_PROMPTS_G,
    REFINEMENT_DENOISE,
    REFINEMENT_STEPS,
    REGIONAL_PROMPT_WEIGHT,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    TILE_BATCH_SIZE,
    TILE_OVERLAP,
    TILE_SIZE,
)
from .sampler_branch import SdxlWorkflowOutputs, add_sampler_branch
from .sampling_controls import SDXL_VISUAL_SAMPLING


@dataclass(frozen=True, slots=True)
class BuiltSdxlAttentionCouplingWorkflow:
    """Return one API graph and exact labeled output-node identities."""

    prompt: dict[str, JsonObject]
    outputs: dict[str, SdxlWorkflowOutputs]

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every class required from clean managed-node discovery."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


def build_sdxl_attention_coupling_workflow(
    *,
    run_id: str,
    checkpoint_name: str,
    mask_names: tuple[str, str],
) -> BuiltSdxlAttentionCouplingWorkflow:
    """Build one source generation and two 1.5x refinement branches."""

    graph = SdxlWorkflowGraph()
    loader = graph.add("CheckpointLoaderSimple", ckpt_name=checkpoint_name)
    positive = _conditioning_batch(
        graph,
        clip=[loader, 1],
        prompts=POSITIVE_PROMPTS,
        global_prompts=POSITIVE_PROMPTS_G,
    )
    negative = _conditioning_batch(
        graph,
        clip=[loader, 1],
        prompts=NEGATIVE_PROMPTS,
        global_prompts=NEGATIVE_PROMPTS_G,
    )
    masks = graph.add(
        "SimpleSyrup.LoadMaskBatch",
        image={"__value__": list(mask_names)},
        channel="red",
    )
    source_latent = graph.add(
        "EmptyLatentImage",
        width=SOURCE_WIDTH,
        height=SOURCE_HEIGHT,
        batch_size=1,
    )
    outputs: dict[str, SdxlWorkflowOutputs] = {}
    full_branch = add_sampler_branch(
        graph,
        mode_id=MODES[0].mode_id,
        node_id=MODES[0].node_id,
        model=[loader, 0],
        positive=positive,
        negative=negative,
        masks=[masks, 0],
        latent=[source_latent, 0],
        vae=[loader, 2],
        run_id=run_id,
        seed=SDXL_VISUAL_SAMPLING.seed,
        steps=SDXL_VISUAL_SAMPLING.steps,
        denoise=1.0,
        regional_prompt_weight=REGIONAL_PROMPT_WEIGHT,
        region_mask_feather=32,
        sampler_inputs={},
    )
    outputs[MODES[0].mode_id] = full_branch.outputs
    upscaled_image = graph.add(
        "ImageScale",
        image=full_branch.decoded,
        upscale_method="lanczos",
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
        crop="disabled",
    )
    upscaled_latent = graph.add(
        "VAEEncode",
        pixels=[upscaled_image, 0],
        vae=[loader, 2],
    )
    tiled_branch = add_sampler_branch(
        graph,
        mode_id=MODES[1].mode_id,
        node_id=MODES[1].node_id,
        model=[loader, 0],
        positive=positive,
        negative=negative,
        masks=[masks, 0],
        latent=[upscaled_latent, 0],
        vae=[loader, 2],
        run_id=run_id,
        seed=SDXL_VISUAL_SAMPLING.seed,
        steps=REFINEMENT_STEPS,
        denoise=REFINEMENT_DENOISE,
        regional_prompt_weight=REGIONAL_PROMPT_WEIGHT,
        region_mask_feather=32,
        sampler_inputs={
            "diffusion_mode": "multidiffusion",
            "latent_tile_width": TILE_SIZE,
            "latent_tile_height": TILE_SIZE,
            "latent_tile_overlap": TILE_OVERLAP,
            "latent_tile_batch_size": TILE_BATCH_SIZE,
        },
    )
    outputs[MODES[1].mode_id] = tiled_branch.outputs
    contextual_branch = add_sampler_branch(
        graph,
        mode_id=MODES[2].mode_id,
        node_id=MODES[2].node_id,
        model=[loader, 0],
        positive=positive,
        negative=negative,
        masks=[masks, 0],
        latent=[upscaled_latent, 0],
        vae=[loader, 2],
        run_id=run_id,
        seed=SDXL_VISUAL_SAMPLING.seed,
        steps=REFINEMENT_STEPS,
        denoise=REFINEMENT_DENOISE,
        regional_prompt_weight=REGIONAL_PROMPT_WEIGHT,
        region_mask_feather=32,
        sampler_inputs={
            "diffusion_mode": "multidiffusion",
            "latent_context_size": TILE_SIZE,
            "latent_context_overlap": TILE_OVERLAP,
            "latent_context_batch_size": TILE_BATCH_SIZE,
            "global_weight": 1.0,
            "global_steps": 1,
            "global_decay": 0.5,
        },
    )
    outputs[MODES[2].mode_id] = contextual_branch.outputs
    return BuiltSdxlAttentionCouplingWorkflow(graph.prompt, outputs)


def _conditioning_batch(
    graph: SdxlWorkflowGraph,
    *,
    clip: list[str | int],
    prompts: tuple[str, ...],
    global_prompts: tuple[str, ...],
) -> list[str | int]:
    """Encode independent SDXL G/L prompts and explicit microconditioning."""

    if len(prompts) != len(global_prompts):
        raise ValueError("SDXL G/L conditioning prompt counts must match.")
    encoded = tuple(
        graph.add(
            "CLIPTextEncodeSDXL",
            clip=clip,
            width=SOURCE_WIDTH,
            height=SOURCE_HEIGHT,
            crop_w=0,
            crop_h=0,
            target_width=TARGET_WIDTH,
            target_height=TARGET_HEIGHT,
            text_g=global_prompt,
            text_l=local_prompt,
        )
        for global_prompt, local_prompt in zip(
            global_prompts,
            prompts,
            strict=True,
        )
    )
    batch = graph.add(
        "SimpleSyrup.ConditioningBatchStart",
        conditioning=[encoded[0], 0],
    )
    for node_id in encoded[1:]:
        batch = graph.add(
            "SimpleSyrup.ConditioningBatchAppend",
            batch=[batch, 0],
            conditioning=[node_id, 0],
        )
    return [batch, 0]
