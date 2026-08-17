# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Assemble shared full, tiled, and Contextual Anima sampler graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tools.anima_attention_coupling_conditioning import (
    AnimaConditioningEvidence,
    AnimaConditioningWorkflow,
)
from tools.anima_latent_source_workflow import (
    AnimaLatentSourceWorkflow,
    EmptyAnimaLatentSourceWorkflow,
)
from tools.anima_workflow_graph import AnimaWorkflowGraph, NodeReference
from tools.comfy_api import JsonObject
from tools.comfy_integration.anima_fixture_selections import (
    ANIMA_DIFFUSION_SELECTION,
    ANIMA_TEXT_ENCODER_SELECTION,
    ANIMA_VAE_SELECTION,
)

SEED = 1_029_384_756
WIDTH = 1024
HEIGHT = 1024
STEPS = 12
SAMPLER = "er_sde"
SCHEDULER = "simple"


class AnimaAttentionCouplingCase(Protocol):
    """Describe case values consumed by the shared graph builder."""

    @property
    def case_id(self) -> str:
        """Return the stable artifact identity."""

    @property
    def cfg(self) -> float:
        """Return the sampler guidance scale."""

    @property
    def feather(self) -> int:
        """Return authored mask feathering in image pixels."""


class AnimaSamplerInputWorkflow(Protocol):
    """Add graph-owned optional inputs for one public sampler invocation."""

    def add(
        self,
        graph: AnimaWorkflowGraph,
        *,
        region_masks: object,
    ) -> dict[str, object]:
        """Return additional sampler inputs backed by nodes in the same graph."""


@dataclass(frozen=True, slots=True)
class ModifiedAnimaModel:
    """Expose explicit MODEL and CLIP links from one modifier graph owner."""

    model: NodeReference
    clip: NodeReference


class AnimaModelModifierWorkflow(Protocol):
    """Transform explicit loader outputs without owning conditioning policy."""

    def add(
        self,
        graph: AnimaWorkflowGraph,
        *,
        model: NodeReference,
        clip: NodeReference,
    ) -> ModifiedAnimaModel:
        """Return modified MODEL and CLIP links backed by graph nodes."""


@dataclass(frozen=True, slots=True)
class ObservedAnimaModel:
    """Expose one pass-through MODEL link and its evidence node identity."""

    model: NodeReference
    evidence_node_id: str


class AnimaModelObserverWorkflow(Protocol):
    """Publish MODEL evidence without owning modification or conditioning."""

    def add(
        self,
        graph: AnimaWorkflowGraph,
        *,
        model: NodeReference,
    ) -> ObservedAnimaModel:
        """Return the pass-through MODEL link and evidence node identity."""


@dataclass(frozen=True, slots=True)
class BuiltAnimaAttentionCouplingWorkflow:
    """Retain one graph and its evidence-producing output identities."""

    prompt: dict[str, JsonObject]
    save_node_id: str
    metrics_node_id: str
    metrics_run_id: str
    diagnostics_node_id: str
    diagnostics_run_id: str
    source_save_node_id: str | None = None
    conditioning_evidence: AnimaConditioningEvidence = AnimaConditioningEvidence()
    conditioning_batch_snapshot_node_id: str | None = None
    conditioning_batch_snapshot_run_id: str | None = None
    modifier_snapshot_node_id: str | None = None

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return all exact node contracts named by the submitted graph."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


class AnimaAttentionCouplingWorkflowBuilder:
    """Translate one case into a loader-to-image public-node workflow."""

    def __init__(
        self,
        *,
        public_node_id: str,
        artifact_phase: str,
        conditioning: AnimaConditioningWorkflow,
        steps: int,
        sampler_inputs: dict[str, object] | None = None,
        sampler_input_workflow: AnimaSamplerInputWorkflow | None = None,
        model_modifier_workflow: AnimaModelModifierWorkflow | None = None,
        model_observer_workflow: AnimaModelObserverWorkflow | None = None,
        latent_source: AnimaLatentSourceWorkflow | None = None,
        denoise: float = 1.0,
    ) -> None:
        """Retain graph collaborators, artifact identity, and spatial controls."""

        if not public_node_id:
            raise ValueError("Anima workflow public node ID must not be empty.")
        if not artifact_phase:
            raise ValueError("Anima workflow artifact phase must not be empty.")
        self._public_node_id = public_node_id
        self._artifact_phase = artifact_phase
        self._conditioning = conditioning
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("Anima workflow steps must be an integer.")
        if steps < 1:
            raise ValueError("Anima workflow steps must be positive.")
        self._steps = steps
        self._sampler_inputs = {} if sampler_inputs is None else sampler_inputs.copy()
        self._sampler_input_workflow = sampler_input_workflow
        self._model_modifier_workflow = model_modifier_workflow
        self._model_observer_workflow = model_observer_workflow
        self._latent_source = (
            EmptyAnimaLatentSourceWorkflow(WIDTH, HEIGHT)
            if latent_source is None
            else latent_source
        )
        if not 0.0 <= denoise <= 1.0:
            raise ValueError("Anima workflow denoise must be between zero and one.")
        self._denoise = denoise

    def build(
        self,
        case: AnimaAttentionCouplingCase,
        *,
        run_id: str,
        mask_names: tuple[str, ...],
    ) -> BuiltAnimaAttentionCouplingWorkflow:
        """Build one fully instrumented public-node workflow."""

        if len(mask_names) != 2:
            raise ValueError("Anima workflows require two ordered mask files.")
        graph = AnimaWorkflowGraph()
        loader = graph.add(
            "SimpleSyrup.SimpleLoadAnima",
            diffusion_model=ANIMA_DIFFUSION_SELECTION,
            quantization="Original",
            diffusion_weight_dtype="default",
            text_encoder=ANIMA_TEXT_ENCODER_SELECTION,
            text_encoder_device="default",
            vae=ANIMA_VAE_SELECTION,
        )
        loaded = ModifiedAnimaModel([loader, 0], [loader, 1])
        modified = (
            loaded
            if self._model_modifier_workflow is None
            else self._model_modifier_workflow.add(
                graph,
                model=loaded.model,
                clip=loaded.clip,
            )
        )
        if not isinstance(modified, ModifiedAnimaModel):
            raise TypeError("Anima model modifier workflow returned an invalid result.")
        observed = (
            None
            if self._model_observer_workflow is None
            else self._model_observer_workflow.add(
                graph,
                model=modified.model,
            )
        )
        if observed is not None and not isinstance(observed, ObservedAnimaModel):
            raise TypeError("Anima model observer workflow returned an invalid result.")
        observed_model = modified.model if observed is None else observed.model
        conditioning = self._conditioning.add(
            graph,
            model=observed_model,
            clip=modified.clip,
        )
        masks = graph.add(
            "SimpleSyrup.LoadMaskBatch",
            image={"__value__": list(mask_names)},
            channel="red",
        )
        source = self._latent_source.add(
            graph,
            model=conditioning.model,
            positive=conditioning.positive,
            negative=conditioning.negative,
            region_masks=[masks, 0],
            vae=[loader, 2],
            seed=SEED,
            steps=self._steps,
            cfg=case.cfg,
            sampler_name=SAMPLER,
            scheduler=SCHEDULER,
            region_mask_feather=case.feather,
        )
        source_save_node_id = (
            None
            if source.source_image is None
            else graph.add(
                "SaveImage",
                images=source.source_image,
                filename_prefix=(
                    f"simple_syrup_{self._artifact_phase}/{run_id}/"
                    f"{case.case_id}-source"
                ),
            )
        )
        metrics_run_id = f"{run_id}:{case.case_id}"
        instrumented = graph.add(
            "SimpleSyrupBenchmark.InstrumentModel",
            model=conditioning.model,
            run_id=metrics_run_id,
        )
        diagnostics_run_id = f"{metrics_run_id}:regional-diagnostics"
        diagnostics_capture = graph.add(
            "SimpleSyrupBenchmark.CaptureRegionalDiagnostics",
            model=[instrumented, 0],
            run_id=diagnostics_run_id,
        )
        graph_sampler_inputs = (
            {}
            if self._sampler_input_workflow is None
            else self._sampler_input_workflow.add(
                graph,
                region_masks=[masks, 0],
            )
        )
        collisions = self._sampler_inputs.keys() & graph_sampler_inputs.keys()
        if collisions:
            raise ValueError(
                f"Anima workflow sampler input owners overlap: {sorted(collisions)!r}."
            )
        sampled = graph.add(
            self._public_node_id,
            model=[diagnostics_capture, 0],
            seed=SEED,
            steps=self._steps,
            cfg=case.cfg,
            sampler_name=SAMPLER,
            scheduler=SCHEDULER,
            positive=conditioning.positive,
            negative=conditioning.negative,
            region_masks=[masks, 0],
            regional_prompt_weight=1.0,
            region_mask_feather=case.feather,
            latent_image=source.latent,
            denoise=self._denoise,
            **self._sampler_inputs,
            **graph_sampler_inputs,
        )
        metrics = graph.add(
            "SimpleSyrupBenchmark.ReadMetrics",
            latent=[sampled, 0],
            run_id=metrics_run_id,
        )
        diagnostics = graph.add(
            "SimpleSyrupBenchmark.ReadRegionalDiagnostics",
            latent=[metrics, 0],
            run_id=diagnostics_run_id,
        )
        decoded = graph.add("VAEDecode", samples=[diagnostics, 0], vae=[loader, 2])
        saved = graph.add(
            "SaveImage",
            images=[decoded, 0],
            filename_prefix=(
                f"simple_syrup_{self._artifact_phase}/{run_id}/{case.case_id}"
            ),
        )
        return BuiltAnimaAttentionCouplingWorkflow(
            graph.prompt,
            saved,
            metrics,
            metrics_run_id,
            diagnostics,
            diagnostics_run_id,
            source_save_node_id,
            conditioning.evidence,
            modifier_snapshot_node_id=(
                None if observed is None else observed.evidence_node_id
            ),
        )
