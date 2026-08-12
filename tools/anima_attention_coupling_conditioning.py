"""Build the accepted scheduled conditioning graph for Anima workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tools.anima_attention_coupling_prompts import (
    NEGATIVE_PROMPT,
    GlobalLora,
    RegionalLora,
    render_regional_positive_prompt,
)
from tools.anima_workflow_graph import AnimaWorkflowGraph, NodeReference


@dataclass(frozen=True, slots=True)
class AnimaConditioningEvidence:
    """Expose optional conditioning-owned managed evidence node identities."""

    expansion_node_id: str | None = None
    snapshot_node_id: str | None = None


@dataclass(frozen=True, slots=True)
class BuiltAnimaConditioning:
    """Expose graph links produced by one conditioning workflow."""

    model: NodeReference
    positive: NodeReference
    negative: NodeReference
    evidence: AnimaConditioningEvidence = AnimaConditioningEvidence()


class AnimaConditioningWorkflow(Protocol):
    """Build conditioning links from explicit MODEL and CLIP references."""

    def add(
        self,
        graph: AnimaWorkflowGraph,
        *,
        model: NodeReference,
        clip: NodeReference,
    ) -> BuiltAnimaConditioning:
        """Add conditioning nodes and return their sampler-facing links."""


class ScheduledRegionalPromptConditioningWorkflow:
    """Build the accepted `[SEP]` Prompt Control regional conditioning graph."""

    def __init__(
        self,
        *,
        global_loras: tuple[GlobalLora, ...],
        regional_loras: tuple[RegionalLora, ...],
    ) -> None:
        """Retain immutable adapter declarations for one managed case."""

        self._global_loras = global_loras
        self._regional_loras = regional_loras

    def add(
        self,
        graph: AnimaWorkflowGraph,
        *,
        model: NodeReference,
        clip: NodeReference,
    ) -> BuiltAnimaConditioning:
        """Apply global LoRAs and add the public scheduled encoder node."""

        for adapter in self._global_loras:
            applied = graph.add(
                "LoraLoaderModelOnly",
                model=model,
                lora_name=adapter.lora_name,
                strength_model=adapter.strength,
            )
            model = [applied, 0]
        encoded = graph.add(
            "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl",
            model=model,
            clip=clip,
            positive_prompt=render_regional_positive_prompt(self._regional_loras),
            negative_prompt=NEGATIVE_PROMPT,
        )
        return BuiltAnimaConditioning([encoded, 0], [encoded, 1], [encoded, 2])
