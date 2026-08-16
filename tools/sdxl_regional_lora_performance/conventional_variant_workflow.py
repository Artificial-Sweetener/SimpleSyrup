# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build image-free conventional single-variant SDXL timing workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tools.comfy_api import JsonObject
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
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    GlobalVisualAdapter,
    RegionalVisualAdapter,
    SdxlVisualCase,
)
from tools.sdxl_attention_coupling_integration.visual_lora_graph import (
    SdxlVisualLoraGraphBuilder,
)

from .steady_state_workflow import BuiltSdxlSteadyStateWorkflow


class ConventionalVariantBranch(StrEnum):
    """Identify one permanently applied conventional adapter branch."""

    LEFT = "conventional-left-variant"
    RIGHT = "conventional-right-variant"


@dataclass(frozen=True, slots=True)
class BuiltConventionalVariantWorkflow:
    """Expose one image-free graph and its synchronized metric terminal."""

    prompt: dict[str, JsonObject]
    metrics_node_id: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every live Comfy node required by this graph."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


@dataclass(frozen=True, slots=True)
class _PreparedConventionalVariantSampling:
    """Retain one native adapter branch before selecting a timing terminal."""

    graph: SdxlWorkflowGraph
    model: NodeReference
    positive: NodeReference
    negative: NodeReference
    latent: NodeReference


def build_conventional_variant_workflow(
    *,
    run_id: str,
    checkpoint_name: str,
    case: SdxlVisualCase,
    branch: ConventionalVariantBranch,
) -> BuiltConventionalVariantWorkflow:
    """Build one ordinary KSampler with exactly one conventional adapter."""

    prepared = _prepare_conventional_variant_sampling(
        checkpoint_name=checkpoint_name,
        case=case,
        branch=branch,
    )
    graph = prepared.graph
    metrics_run_id = f"{run_id}:{branch.value}:metrics"
    instrumented = graph.add(
        "SimpleSyrupBenchmark.InstrumentModel",
        model=prepared.model,
        run_id=metrics_run_id,
    )
    sampled = _add_sampler(prepared, model=[instrumented, 0])
    metrics = graph.add(
        "SimpleSyrupBenchmark.ReadMetrics",
        latent=[sampled, 0],
        run_id=metrics_run_id,
    )
    return BuiltConventionalVariantWorkflow(graph.prompt, metrics)


def build_conventional_variant_steady_state_workflow(
    *,
    checkpoint_name: str,
    case: SdxlVisualCase,
    branch: ConventionalVariantBranch,
) -> BuiltSdxlSteadyStateWorkflow:
    """Build one cache-realistic conventional adapter branch."""

    prepared = _prepare_conventional_variant_sampling(
        checkpoint_name=checkpoint_name,
        case=case,
        branch=branch,
    )
    sampler = _add_sampler(prepared, model=prepared.model)
    completion = prepared.graph.add(
        "SimpleSyrupBenchmark.CompleteLatent",
        latent=[sampler, 0],
    )
    return BuiltSdxlSteadyStateWorkflow(
        prompt=prepared.graph.prompt,
        sampler_node_id=sampler,
        completion_node_id=completion,
    )


def _prepare_conventional_variant_sampling(
    *,
    checkpoint_name: str,
    case: SdxlVisualCase,
    branch: ConventionalVariantBranch,
) -> _PreparedConventionalVariantSampling:
    """Build the shared native adapter, conditioning, and latent graph."""

    if not isinstance(case, SdxlVisualCase):
        raise TypeError("Conventional variant timing requires an SDXL case.")
    if not isinstance(branch, ConventionalVariantBranch):
        raise TypeError("Conventional variant timing branch is invalid.")
    adapter = _adapter(case, branch)
    if adapter.model_strength != adapter.clip_strength:
        raise ValueError("Conventional variant requires equal model and CLIP strength.")
    graph = SdxlWorkflowGraph()
    loader = graph.add("CheckpointLoaderSimple", ckpt_name=checkpoint_name)
    loaded = SdxlVisualLoraGraphBuilder().load_global(
        graph,
        model=[loader, 0],
        clip=[loader, 1],
        adapters=(GlobalVisualAdapter(adapter.lora_name, adapter.model_strength),),
    )
    positive = _encode(
        graph,
        loaded.clip,
        _join(case.base_positive_g, case.left_g, case.right_g),
        _join(case.base_positive_l, case.left_l, case.right_l),
    )
    negative = _encode(
        graph,
        loaded.clip,
        _join(case.base_negative_g, case.left_negative_g, case.right_negative_g),
        _join(case.base_negative_l, case.left_negative_l, case.right_negative_l),
    )
    latent = graph.add(
        "EmptyLatentImage",
        width=SOURCE_WIDTH,
        height=SOURCE_HEIGHT,
        batch_size=1,
    )
    return _PreparedConventionalVariantSampling(
        graph=graph,
        model=loaded.model,
        positive=positive,
        negative=negative,
        latent=[latent, 0],
    )


def _add_sampler(
    prepared: _PreparedConventionalVariantSampling,
    *,
    model: NodeReference,
) -> str:
    """Append the locked ordinary sampler to one prepared branch."""

    return prepared.graph.add(
        "KSampler",
        model=model,
        seed=SDXL_VISUAL_SAMPLING.seed,
        steps=SDXL_VISUAL_SAMPLING.steps,
        cfg=SDXL_VISUAL_SAMPLING.cfg,
        sampler_name=SDXL_VISUAL_SAMPLING.sampler,
        scheduler=SDXL_VISUAL_SAMPLING.scheduler,
        positive=prepared.positive,
        negative=prepared.negative,
        latent_image=prepared.latent,
        denoise=1.0,
    )


def _adapter(
    case: SdxlVisualCase,
    branch: ConventionalVariantBranch,
) -> RegionalVisualAdapter:
    """Return one exact single-adapter branch declaration."""

    selected = (
        case.left_adapters
        if branch is ConventionalVariantBranch.LEFT
        else case.right_adapters
    )
    if len(selected) != 1:
        raise ValueError("Conventional variant requires one adapter on each side.")
    return selected[0]


def _encode(
    graph: SdxlWorkflowGraph,
    clip: NodeReference,
    text_g: str,
    text_l: str,
) -> NodeReference:
    """Encode one ordinary native SDXL dual-encoder conditioning."""

    node = graph.add(
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
    return [node, 0]


def _join(*segments: str) -> str:
    """Join only non-empty authored prompt segments in order."""

    return ", ".join(segment.strip() for segment in segments if segment.strip())
