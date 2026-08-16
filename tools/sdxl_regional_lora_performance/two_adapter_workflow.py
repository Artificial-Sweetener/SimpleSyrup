# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build image-free matched global and two-region SDXL LoRA workflows."""

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
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    GlobalVisualAdapter,
    SdxlVisualCase,
)
from tools.sdxl_attention_coupling_integration.visual_lora_graph import (
    SdxlVisualLoraGraphBuilder,
)

from .sampling_workflow import (
    PreparedSdxlSampling,
    add_sdxl_sampler,
    prepare_regional_sampling,
)
from .steady_state_workflow import BuiltSdxlSteadyStateWorkflow


class TwoAdapterPerformanceMode(StrEnum):
    """Identify ordinary-global or two-region adapter placement."""

    GLOBAL_REFERENCE = "global-two-lora-reference"
    REGIONAL = "two-regional-loras"


@dataclass(frozen=True, slots=True)
class BuiltTwoAdapterPerformanceWorkflow:
    """Expose one image-free graph and its exact metrics terminal."""

    prompt: dict[str, JsonObject]
    metrics_node_id: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every live Comfy node required by this graph."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


def build_two_adapter_performance_workflow(
    *,
    run_id: str,
    checkpoint_name: str,
    mask_names: tuple[str, str],
    case: SdxlVisualCase,
    mode: TwoAdapterPerformanceMode,
) -> BuiltTwoAdapterPerformanceWorkflow:
    """Build one matched 30-step sampler without decode or artifact work."""

    prepared = _prepare_two_adapter_sampling(
        checkpoint_name=checkpoint_name,
        mask_names=mask_names,
        case=case,
        mode=mode,
    )
    graph = prepared.graph
    metrics_run_id = f"{run_id}:{mode.value}:metrics"
    instrumented = graph.add(
        "SimpleSyrupBenchmark.InstrumentModel",
        model=prepared.model,
        run_id=metrics_run_id,
    )
    sampled = add_sdxl_sampler(prepared, model=[instrumented, 0])
    metrics = graph.add(
        "SimpleSyrupBenchmark.ReadMetrics",
        latent=[sampled, 0],
        run_id=metrics_run_id,
    )
    return BuiltTwoAdapterPerformanceWorkflow(graph.prompt, metrics)


def build_two_adapter_steady_state_workflow(
    *,
    checkpoint_name: str,
    mask_names: tuple[str, str],
    case: SdxlVisualCase,
    mode: TwoAdapterPerformanceMode,
) -> BuiltSdxlSteadyStateWorkflow:
    """Build one cache-realistic graph with no MODEL-cloning timing probe."""

    prepared = _prepare_two_adapter_sampling(
        checkpoint_name=checkpoint_name,
        mask_names=mask_names,
        case=case,
        mode=mode,
    )
    sampler = add_sdxl_sampler(prepared, model=prepared.model)
    completion = prepared.graph.add(
        "SimpleSyrupBenchmark.CompleteLatent",
        latent=[sampler, 0],
    )
    return BuiltSdxlSteadyStateWorkflow(
        prompt=prepared.graph.prompt,
        sampler_node_id=sampler,
        completion_node_id=completion,
    )


def _prepare_two_adapter_sampling(
    *,
    checkpoint_name: str,
    mask_names: tuple[str, str],
    case: SdxlVisualCase,
    mode: TwoAdapterPerformanceMode,
) -> PreparedSdxlSampling:
    """Build the shared model, conditioning, latent, and regional controls."""

    _validate_case(case)
    if not isinstance(mode, TwoAdapterPerformanceMode):
        raise TypeError("Two-adapter performance mode is invalid.")
    if mode is TwoAdapterPerformanceMode.REGIONAL:
        return prepare_regional_sampling(
            checkpoint_name=checkpoint_name,
            mask_names=mask_names,
            case=case,
        )
    graph = SdxlWorkflowGraph()
    loader = graph.add("CheckpointLoaderSimple", ckpt_name=checkpoint_name)
    model, positive, negative = _global_conditioning(
        graph,
        model=[loader, 0],
        clip=[loader, 1],
        case=case,
    )
    latent_node = graph.add(
        "EmptyLatentImage",
        width=SOURCE_WIDTH,
        height=SOURCE_HEIGHT,
        batch_size=1,
    )
    return PreparedSdxlSampling(
        graph=graph,
        model=model,
        positive=positive,
        negative=negative,
        latent=[latent_node, 0],
        sampler_class="KSampler",
        sampler_extras={},
    )


def _global_conditioning(
    graph: SdxlWorkflowGraph,
    *,
    model: NodeReference,
    clip: NodeReference,
    case: SdxlVisualCase,
) -> tuple[NodeReference, NodeReference, NodeReference]:
    """Apply both adapters conventionally and encode one combined prompt."""

    adapters = tuple(
        GlobalVisualAdapter(adapter.lora_name, adapter.model_strength)
        for adapter in (*case.left_adapters, *case.right_adapters)
    )
    loaded = SdxlVisualLoraGraphBuilder().load_global(
        graph,
        model=model,
        clip=clip,
        adapters=adapters,
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
    return loaded.model, positive, negative


def _encode(
    graph: SdxlWorkflowGraph,
    clip: NodeReference,
    text_g: str,
    text_l: str,
) -> NodeReference:
    """Encode one native SDXL dual-encoder prompt with locked geometry."""

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


def _validate_case(case: SdxlVisualCase) -> None:
    """Require exactly one full-strength adapter on each regional side."""

    if not isinstance(case, SdxlVisualCase):
        raise TypeError("Two-adapter performance requires an SDXL visual case.")
    if len(case.left_adapters) != 1 or len(case.right_adapters) != 1:
        raise ValueError("Two-adapter performance requires one adapter per side.")
    for adapter in (*case.left_adapters, *case.right_adapters):
        if adapter.model_strength != adapter.clip_strength:
            raise ValueError(
                "Matched global LoRA comparison requires equal model and CLIP strength."
            )
