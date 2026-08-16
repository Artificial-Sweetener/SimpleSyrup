# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build deterministic reference and candidate SDXL parity graphs."""

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

from .cases import SdxlAttentionCoupleParityCase
from .global_style import (
    ParityGlobalStyle,
    append_base_style,
    load_parity_global_style,
)
from .ownership import GLOBAL_OWNERSHIP_STRENGTH, REGIONAL_OWNERSHIP_STRENGTH


class ParityBackend(StrEnum):
    """Name the sole changed axis in the controlled comparison."""

    REFERENCE = "reference"
    CANDIDATE = "candidate"


@dataclass(frozen=True, slots=True)
class BuiltParityWorkflow:
    """Expose one backend graph and its sole terminal image node."""

    backend: ParityBackend
    prompt: dict[str, JsonObject]
    save_node_id: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every live Comfy node required by this graph."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


@dataclass(frozen=True, slots=True)
class _SharedGraphInputs:
    """Retain identical model, conditioning, and latent graph references."""

    model: NodeReference
    clip: NodeReference
    vae: NodeReference
    base_positive: NodeReference
    left_positive: NodeReference
    right_positive: NodeReference
    base_negative: NodeReference
    left_negative: NodeReference
    right_negative: NodeReference
    latent: NodeReference


def build_parity_workflow(
    *,
    backend: ParityBackend,
    run_id: str,
    checkpoint_name: str,
    mask_names: tuple[str, str],
    case: SdxlAttentionCoupleParityCase,
    global_style: ParityGlobalStyle | None = None,
) -> BuiltParityWorkflow:
    """Build one graph differing only at regional attention execution."""

    graph = SdxlWorkflowGraph()
    shared = _add_shared_inputs(
        graph,
        checkpoint_name=checkpoint_name,
        case=case,
        global_style=global_style,
    )
    if backend is ParityBackend.REFERENCE:
        sampled = _add_reference_sampler(graph, shared=shared, mask_names=mask_names)
    elif backend is ParityBackend.CANDIDATE:
        sampled = _add_candidate_sampler(graph, shared=shared, mask_names=mask_names)
    else:
        raise ValueError(f"Unsupported parity backend: {backend!r}")
    decoded = graph.add("VAEDecode", samples=sampled, vae=shared.vae)
    saved = graph.add(
        "SaveImage",
        images=[decoded, 0],
        filename_prefix=f"simple_syrup_ra01/{run_id}/{backend.value}",
    )
    return BuiltParityWorkflow(backend, graph.prompt, saved)


def _add_shared_inputs(
    graph: SdxlWorkflowGraph,
    *,
    checkpoint_name: str,
    case: SdxlAttentionCoupleParityCase,
    global_style: ParityGlobalStyle | None,
) -> _SharedGraphInputs:
    """Add the byte-for-byte equal controls used by both backend graphs."""

    loader = graph.add("CheckpointLoaderSimple", ckpt_name=checkpoint_name)
    model: NodeReference = [loader, 0]
    clip: NodeReference = [loader, 1]
    base_positive_g = case.base_positive_g
    base_positive_l = case.base_positive_l
    if global_style is not None:
        loaded = load_parity_global_style(
            graph,
            style=global_style,
            model=model,
            clip=clip,
        )
        model = loaded.model
        clip = loaded.clip
        base_positive_g = append_base_style(base_positive_g, global_style.prompt_g)
        base_positive_l = append_base_style(base_positive_l, global_style.prompt_l)
    base_positive = _encode(
        graph, clip=clip, text_g=base_positive_g, text_l=base_positive_l
    )
    left_positive = _encode(
        graph, clip=clip, text_g=case.left_positive_g, text_l=case.left_positive_l
    )
    right_positive = _encode(
        graph, clip=clip, text_g=case.right_positive_g, text_l=case.right_positive_l
    )
    base_negative = _encode(
        graph, clip=clip, text_g=case.base_negative_g, text_l=case.base_negative_l
    )
    left_negative = _encode(
        graph, clip=clip, text_g=case.left_negative_g, text_l=case.left_negative_l
    )
    right_negative = _encode(
        graph, clip=clip, text_g=case.right_negative_g, text_l=case.right_negative_l
    )
    latent = graph.add(
        "EmptyLatentImage",
        width=SOURCE_WIDTH,
        height=SOURCE_HEIGHT,
        batch_size=1,
    )
    return _SharedGraphInputs(
        model,
        clip,
        [loader, 2],
        base_positive,
        left_positive,
        right_positive,
        base_negative,
        left_negative,
        right_negative,
        [latent, 0],
    )


def _add_reference_sampler(
    graph: SdxlWorkflowGraph,
    *,
    shared: _SharedGraphInputs,
    mask_names: tuple[str, str],
) -> NodeReference:
    """Add the installed known-good Attention Couple model patch and sampler."""

    left_mask = graph.add("LoadImageMask", image=mask_names[0], channel="red")
    right_mask = graph.add("LoadImageMask", image=mask_names[1], channel="red")
    full_mask = graph.add(
        "SolidMask",
        value=GLOBAL_OWNERSHIP_STRENGTH,
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
    )
    regional_strength = graph.add(
        "SolidMask",
        value=REGIONAL_OWNERSHIP_STRENGTH,
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
    )
    weighted_left = graph.add(
        "MaskComposite",
        destination=[regional_strength, 0],
        source=[left_mask, 0],
        x=0,
        y=0,
        operation="multiply",
    )
    weighted_right = graph.add(
        "MaskComposite",
        destination=[regional_strength, 0],
        source=[right_mask, 0],
        x=0,
        y=0,
        operation="multiply",
    )
    patched = graph.add(
        "AttentionCouplePPM",
        model=shared.model,
        base_cond=shared.base_positive,
        base_mask=[full_mask, 0],
        cond_1=shared.left_positive,
        mask_1=[weighted_left, 0],
        cond_2=shared.right_positive,
        mask_2=[weighted_right, 0],
    )
    sampled = graph.add(
        "KSampler",
        model=[patched, 0],
        **_sampler_controls(shared),
    )
    return [sampled, 0]


def _add_candidate_sampler(
    graph: SdxlWorkflowGraph,
    *,
    shared: _SharedGraphInputs,
    mask_names: tuple[str, str],
) -> NodeReference:
    """Add the current SimpleSyrup standard-UNet Attention Coupling sampler."""

    positive = _conditioning_batch(
        graph,
        (shared.base_positive, shared.left_positive, shared.right_positive),
    )
    negative = _conditioning_batch(
        graph,
        (shared.base_negative, shared.left_negative, shared.right_negative),
    )
    masks = graph.add(
        "SimpleSyrup.LoadMaskBatch",
        image={"__value__": list(mask_names)},
        channel="red",
    )
    sampled = graph.add(
        "SimpleSyrup.KSamplerAttentionCoupling",
        model=shared.model,
        positive=positive,
        negative=negative,
        region_masks=[masks, 0],
        regional_prompt_weight=REGIONAL_OWNERSHIP_STRENGTH,
        region_mask_feather=0,
        **_sampling_values(shared),
    )
    return [sampled, 0]


def _conditioning_batch(
    graph: SdxlWorkflowGraph,
    entries: tuple[NodeReference, ...],
) -> NodeReference:
    """Pack one global-first ordered conditioning batch."""

    current = graph.add(
        "SimpleSyrup.ConditioningBatchStart",
        conditioning=entries[0],
    )
    for entry in entries[1:]:
        current = graph.add(
            "SimpleSyrup.ConditioningBatchAppend",
            batch=[current, 0],
            conditioning=entry,
        )
    return [current, 0]


def _sampler_controls(shared: _SharedGraphInputs) -> dict[str, object]:
    """Return ordinary KSampler controls with the shared base conditioning."""

    return {
        "positive": shared.base_positive,
        "negative": shared.base_negative,
        **_sampling_values(shared),
    }


def _sampling_values(shared: _SharedGraphInputs) -> dict[str, object]:
    """Return the exact sampling values common to both backends."""

    return {
        "seed": SDXL_VISUAL_SAMPLING.seed,
        "steps": SDXL_VISUAL_SAMPLING.steps,
        "cfg": SDXL_VISUAL_SAMPLING.cfg,
        "sampler_name": SDXL_VISUAL_SAMPLING.sampler,
        "scheduler": SDXL_VISUAL_SAMPLING.scheduler,
        "latent_image": shared.latent,
        "denoise": 1.0,
    }


def _encode(
    graph: SdxlWorkflowGraph,
    *,
    clip: NodeReference,
    text_g: str,
    text_l: str,
) -> NodeReference:
    """Encode one native SDXL G/L pair at the source canvas geometry."""

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
