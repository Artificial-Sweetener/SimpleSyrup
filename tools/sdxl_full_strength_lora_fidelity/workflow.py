# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build fair global and all-one regional SDXL character-LoRA workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.graph import (
    NodeReference,
    SdxlWorkflowGraph,
)
from tools.sdxl_attention_coupling_integration.sampling_controls import (
    SDXL_VISUAL_SAMPLING,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    GlobalVisualAdapter,
    RegionalVisualAdapter,
)
from tools.sdxl_attention_coupling_integration.visual_lora_graph import (
    SdxlVisualLoraGraphBuilder,
)

from .cases import FidelityExecutionMode, FullStrengthFidelityCase


@dataclass(frozen=True, slots=True)
class BuiltFullStrengthFidelityWorkflow:
    """Expose one graph and its terminal image/metrics/diagnostic nodes."""

    prompt: dict[str, JsonObject]
    save_node_id: str
    metrics_node_id: str
    diagnostics_node_id: str | None

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every live Comfy node required by this workflow."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


def build_fidelity_workflow(
    *,
    run_id: str,
    checkpoint_name: str,
    mask_name: str,
    base_positive_g: str,
    base_positive_l: str,
    negative_g: str,
    negative_l: str,
    case: FullStrengthFidelityCase,
) -> BuiltFullStrengthFidelityWorkflow:
    """Build one placement-only comparison with identical sampling controls."""

    graph = SdxlWorkflowGraph()
    loader = graph.add("CheckpointLoaderSimple", ckpt_name=checkpoint_name)
    model: NodeReference = [loader, 0]
    clip: NodeReference = [loader, 1]
    if case.mode is FidelityExecutionMode.GLOBAL_REFERENCE:
        return _build_global(
            graph,
            run_id=run_id,
            case=case,
            model=model,
            clip=clip,
            vae=[loader, 2],
            positive_g=_join(base_positive_g, case.character.prompt_g),
            positive_l=_join(base_positive_l, case.character.prompt_l),
            negative_g=negative_g,
            negative_l=negative_l,
        )
    if case.mode is FidelityExecutionMode.REGIONAL_ALL_ONE:
        return _build_regional(
            graph,
            run_id=run_id,
            case=case,
            model=model,
            clip=clip,
            vae=[loader, 2],
            mask_name=mask_name,
            base_positive_g=base_positive_g,
            base_positive_l=base_positive_l,
            negative_g=negative_g,
            negative_l=negative_l,
        )
    raise ValueError(f"Unsupported fidelity execution mode: {case.mode!r}")


def _build_global(
    graph: SdxlWorkflowGraph,
    *,
    run_id: str,
    case: FullStrengthFidelityCase,
    model: NodeReference,
    clip: NodeReference,
    vae: NodeReference,
    positive_g: str,
    positive_l: str,
    negative_g: str,
    negative_l: str,
) -> BuiltFullStrengthFidelityWorkflow:
    """Build the ordinary global-LoRA reference branch."""

    loaded = SdxlVisualLoraGraphBuilder().load_global(
        graph,
        model=model,
        clip=clip,
        adapters=(GlobalVisualAdapter(case.character.adapter_name, 1.0),),
    )
    positive = _encode(graph, loaded.clip, positive_g, positive_l)
    negative = _encode(graph, loaded.clip, negative_g, negative_l)
    latent = graph.add("EmptyLatentImage", width=1024, height=1024, batch_size=1)
    metrics_run_id = f"{run_id}:{case.case_id}:metrics"
    instrumented = graph.add(
        "SimpleSyrupBenchmark.InstrumentModel",
        model=loaded.model,
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
    decoded = graph.add("VAEDecode", samples=[metrics, 0], vae=vae)
    saved = _save(graph, decoded, run_id, case)
    return BuiltFullStrengthFidelityWorkflow(graph.prompt, saved, metrics, None)


def _build_regional(
    graph: SdxlWorkflowGraph,
    *,
    run_id: str,
    case: FullStrengthFidelityCase,
    model: NodeReference,
    clip: NodeReference,
    vae: NodeReference,
    mask_name: str,
    base_positive_g: str,
    base_positive_l: str,
    negative_g: str,
    negative_l: str,
) -> BuiltFullStrengthFidelityWorkflow:
    """Build the single all-one regional-LoRA comparison branch."""

    base_positive = _encode(graph, clip, base_positive_g, base_positive_l)
    base_negative = _encode(graph, clip, negative_g, negative_l)
    adapter = RegionalVisualAdapter(
        case.character.adapter_name,
        case.model_strength,
        case.clip_strength,
        case.schedule,
    )
    hooks, identities = SdxlVisualLoraGraphBuilder().regional_hooks(graph, (adapter,))
    if hooks is None:
        raise RuntimeError("Full-strength regional fidelity hook is unavailable.")
    labeled = graph.add(
        "SimpleSyrup.LabelRegionalLoraHooks",
        hooks=hooks,
        adapter_identities_json=json.dumps(identities, separators=(",", ":")),
    )
    prepared = graph.add(
        "SimpleSyrup.PrepareRegionalLoraHooks",
        clip=clip,
        hooks=[labeled, 0],
    )
    regional_positive = _encode(
        graph,
        [prepared, 0],
        _join(base_positive_g, case.character.prompt_g),
        _join(base_positive_l, case.character.prompt_l),
    )
    regional_negative = _encode(
        graph,
        [prepared, 0],
        negative_g,
        negative_l,
    )
    regional_positive = _attach(graph, regional_positive, [prepared, 1])
    regional_negative = _attach(graph, regional_negative, [prepared, 1])
    positive = _pack(graph, base_positive, regional_positive)
    negative = _pack(graph, base_negative, regional_negative)
    mask = graph.add(
        "SimpleSyrup.LoadMaskBatch",
        image={"__value__": [mask_name]},
        channel="red",
    )
    latent = graph.add("EmptyLatentImage", width=1024, height=1024, batch_size=1)
    metrics_run_id = f"{run_id}:{case.case_id}:metrics"
    instrumented = graph.add(
        "SimpleSyrupBenchmark.InstrumentModel",
        model=model,
        run_id=metrics_run_id,
    )
    sampled = graph.add(
        "SimpleSyrup.KSamplerAttentionCoupling",
        model=[instrumented, 0],
        seed=SDXL_VISUAL_SAMPLING.seed,
        steps=SDXL_VISUAL_SAMPLING.steps,
        cfg=SDXL_VISUAL_SAMPLING.cfg,
        sampler_name=SDXL_VISUAL_SAMPLING.sampler,
        scheduler=SDXL_VISUAL_SAMPLING.scheduler,
        positive=positive,
        negative=negative,
        region_masks=[mask, 0],
        regional_prompt_weight=1.0,
        region_mask_feather=0,
        latent_image=[latent, 0],
        denoise=1.0,
    )
    metrics = graph.add(
        "SimpleSyrupBenchmark.ReadMetrics",
        latent=[sampled, 0],
        run_id=metrics_run_id,
    )
    decoded = graph.add("VAEDecode", samples=[metrics, 0], vae=vae)
    saved = _save(graph, decoded, run_id, case)
    return BuiltFullStrengthFidelityWorkflow(
        graph.prompt,
        saved,
        metrics,
        None,
    )


def _encode(
    graph: SdxlWorkflowGraph,
    clip: NodeReference,
    text_g: str,
    text_l: str,
) -> NodeReference:
    """Encode one native SDXL G/L prompt with exact microconditioning."""

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


def _attach(
    graph: SdxlWorkflowGraph,
    conditioning: NodeReference,
    hooks: NodeReference,
) -> NodeReference:
    """Attach one regional hook to its matching conditioning."""

    node = graph.add(
        "ConditioningSetProperties",
        cond_NEW=conditioning,
        hooks=hooks,
        strength=1.0,
        set_cond_area="default",
    )
    return [node, 0]


def _pack(
    graph: SdxlWorkflowGraph,
    base: NodeReference,
    regional: NodeReference,
) -> NodeReference:
    """Pack one global base followed by one regional conditioning."""

    started = graph.add("SimpleSyrup.ConditioningBatchStart", conditioning=base)
    appended = graph.add(
        "SimpleSyrup.ConditioningBatchAppend",
        batch=[started, 0],
        conditioning=regional,
    )
    return [appended, 0]


def _save(
    graph: SdxlWorkflowGraph,
    decoded: str,
    run_id: str,
    case: FullStrengthFidelityCase,
) -> str:
    """Save one exact case image under its managed run identity."""

    return graph.add(
        "SaveImage",
        images=[decoded, 0],
        filename_prefix=(
            f"simple_syrup_full_strength_fidelity/{run_id}/{case.case_id}/full"
        ),
    )


def _join(base: str, addition: str) -> str:
    """Join two non-empty prompt fragments without changing their weights."""

    if not base.strip() or not addition.strip():
        raise ValueError("Full-strength fidelity prompts must be non-empty.")
    return f"{base}, {addition}"
