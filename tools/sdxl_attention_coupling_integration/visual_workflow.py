# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build one managed native-SDXL U11 visual acceptance workflow."""

from __future__ import annotations

from dataclasses import dataclass

from tools.comfy_api import JsonObject

from .graph import SdxlWorkflowGraph
from .matrix import (
    MODES,
    REFINEMENT_DENOISE,
    REFINEMENT_STEPS,
    SOURCE_HEIGHT,
    SOURCE_STEPS,
    SOURCE_WIDTH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    TILE_BATCH_SIZE,
    TILE_OVERLAP,
    TILE_SIZE,
)
from .sampler_branch import SdxlWorkflowOutputs, add_sampler_branch
from .visual_cases import SdxlVisualCase, VisualMode
from .visual_conditioning import SdxlVisualConditioningBuilder


@dataclass(frozen=True, slots=True)
class SdxlVisualWorkflowOutput:
    """Retain user label, spatial mode, and terminal node identities."""

    artifact_id: str
    case_id: str
    label: str
    mode: VisualMode
    outputs: SdxlWorkflowOutputs


@dataclass(frozen=True, slots=True)
class BuiltSdxlVisualWorkflow:
    """Expose one case graph and its exact labeled terminal outputs."""

    prompt: dict[str, JsonObject]
    outputs: tuple[SdxlVisualWorkflowOutput, ...]

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every live Comfy node required by this graph."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


def build_sdxl_visual_workflow(
    *,
    run_id: str,
    checkpoint_name: str,
    mask_names: tuple[str, str],
    case: SdxlVisualCase,
) -> BuiltSdxlVisualWorkflow:
    """Build one full source and only the case's declared refinement modes."""

    graph = SdxlWorkflowGraph()
    loader = graph.add("CheckpointLoaderSimple", ckpt_name=checkpoint_name)
    conditioning = SdxlVisualConditioningBuilder().build(
        graph,
        case=case,
        model=[loader, 0],
        clip=[loader, 1],
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
    case_run_id = f"{run_id}:{case.case_id}"
    full = add_sampler_branch(
        graph,
        mode_id=VisualMode.FULL.value,
        node_id=MODES[0].node_id,
        model=conditioning.model,
        positive=conditioning.positive,
        negative=conditioning.negative,
        masks=[masks, 0],
        latent=[source_latent, 0],
        vae=[loader, 2],
        run_id=case_run_id,
        steps=SOURCE_STEPS,
        denoise=1.0,
        sampler_inputs={},
        filename_prefix=_filename_prefix(run_id, case.case_id, VisualMode.FULL),
    )
    output_records = [
        _output_record(case, VisualMode.FULL, full.outputs),
    ]
    refinement_modes = tuple(mode for mode in case.modes if mode is not VisualMode.FULL)
    if refinement_modes:
        upscaled_image = graph.add(
            "ImageScale",
            image=full.decoded,
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
        for mode in refinement_modes:
            branch = add_sampler_branch(
                graph,
                mode_id=mode.value,
                node_id=_node_id(mode),
                model=conditioning.model,
                positive=conditioning.positive,
                negative=conditioning.negative,
                masks=[masks, 0],
                latent=[upscaled_latent, 0],
                vae=[loader, 2],
                run_id=case_run_id,
                steps=REFINEMENT_STEPS,
                denoise=REFINEMENT_DENOISE,
                sampler_inputs=_sampler_inputs(mode),
                filename_prefix=_filename_prefix(run_id, case.case_id, mode),
            )
            output_records.append(_output_record(case, mode, branch.outputs))
    if tuple(record.mode for record in output_records) != case.modes:
        raise ValueError("U11 workflow output order must match the case declaration.")
    return BuiltSdxlVisualWorkflow(graph.prompt, tuple(output_records))


def _output_record(
    case: SdxlVisualCase,
    mode: VisualMode,
    outputs: SdxlWorkflowOutputs,
) -> SdxlVisualWorkflowOutput:
    """Bind one terminal output to its durable case and mode identity."""

    return SdxlVisualWorkflowOutput(
        f"{case.case_id}--{mode.value}",
        case.case_id,
        case.label,
        mode,
        outputs,
    )


def _node_id(mode: VisualMode) -> str:
    """Return the exact public sampler node for one refinement geometry."""

    nodes = {
        VisualMode.TILED: MODES[1].node_id,
        VisualMode.CONTEXTUAL: MODES[2].node_id,
    }
    try:
        return nodes[mode]
    except KeyError as error:
        raise ValueError(f"Unsupported U11 refinement mode: {mode!r}") from error


def _sampler_inputs(mode: VisualMode) -> dict[str, object]:
    """Return exact public inputs for one declared refinement mode."""

    if mode is VisualMode.TILED:
        return {
            "diffusion_mode": "multidiffusion",
            "latent_tile_width": TILE_SIZE,
            "latent_tile_height": TILE_SIZE,
            "latent_tile_overlap": TILE_OVERLAP,
            "latent_tile_batch_size": TILE_BATCH_SIZE,
        }
    if mode is VisualMode.CONTEXTUAL:
        return {
            "diffusion_mode": "multidiffusion",
            "latent_context_size": TILE_SIZE,
            "latent_context_overlap": TILE_OVERLAP,
            "latent_context_batch_size": TILE_BATCH_SIZE,
            "global_weight": 1.0,
            "global_steps": 1,
            "global_decay": 0.5,
        }
    raise ValueError(f"Unsupported U11 refinement mode: {mode!r}")


def _filename_prefix(run_id: str, case_id: str, mode: VisualMode) -> str:
    """Return one stable SaveImage prefix for managed history and cleanup."""

    return f"simple_syrup_u11/{run_id}/{case_id}/{mode.value}"
