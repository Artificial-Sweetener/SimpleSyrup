# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the shared managed Anima model-modifier graph boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from tools.anima_attention_coupling_conditioning import BuiltAnimaConditioning
from tools.anima_attention_coupling_workflow import (
    AnimaAttentionCouplingWorkflowBuilder,
    AnimaModelModifierWorkflow,
    AnimaModelObserverWorkflow,
    ModifiedAnimaModel,
    ObservedAnimaModel,
)
from tools.anima_latent_source_workflow import EmptyAnimaLatentSourceWorkflow
from tools.anima_workflow_graph import AnimaWorkflowGraph, NodeReference


@dataclass(frozen=True, slots=True)
class _Case:
    """Provide the exact values consumed by the shared workflow builder."""

    case_id: str = "modifier-boundary"
    cfg: float = 1.0
    feather: int = 0


class _NegPipModifier:
    """Build one public modifier over explicit loader references."""

    def add(
        self,
        graph: AnimaWorkflowGraph,
        *,
        model: NodeReference,
        clip: NodeReference,
    ) -> ModifiedAnimaModel:
        """Return both outputs of the public NegPiP node."""

        node = graph.add("CLIPNegPip", model=model, clip=clip)
        return ModifiedAnimaModel([node, 0], [node, 1])


class _Conditioning:
    """Record explicit MODEL and CLIP links without reading a loader node."""

    def add(
        self,
        graph: AnimaWorkflowGraph,
        *,
        model: NodeReference,
        clip: NodeReference,
    ) -> BuiltAnimaConditioning:
        """Encode two placeholders and preserve the supplied MODEL link."""

        positive = graph.add("CLIPTextEncode", clip=clip, text="positive")
        negative = graph.add("CLIPTextEncode", clip=clip, text="negative")
        return BuiltAnimaConditioning(model, [positive, 0], [negative, 0])


class _Observer:
    """Publish one model snapshot without owning modification policy."""

    def add(
        self,
        graph: AnimaWorkflowGraph,
        *,
        model: NodeReference,
    ) -> ObservedAnimaModel:
        """Return the pass-through MODEL link and evidence node identity."""

        node = graph.add(
            "SimpleSyrupBenchmark.SnapshotModelModifier",
            model=model,
            run_id="run:case",
        )
        return ObservedAnimaModel([node, 0], node)


def test_shared_builder_routes_modifier_outputs_without_loader_reach_in() -> None:
    """Keep model modification separate from conditioning graph ownership."""

    built = AnimaAttentionCouplingWorkflowBuilder(
        public_node_id="SimpleSyrup.KSamplerAttentionCoupling",
        artifact_phase="p9.7",
        conditioning=_Conditioning(),
        model_modifier_workflow=_NegPipModifier(),
        model_observer_workflow=_Observer(),
        steps=1,
        latent_source=EmptyAnimaLatentSourceWorkflow(64, 64),
    ).build(
        _Case(),
        run_id="run",
        mask_names=("left.png", "right.png"),
    )

    modifier_id = _only_node_id(built.prompt, "CLIPNegPip")
    observer_id = _only_node_id(
        built.prompt,
        "SimpleSyrupBenchmark.SnapshotModelModifier",
    )
    modifier = built.prompt[modifier_id]
    loader_id = _only_node_id(built.prompt, "SimpleSyrup.SimpleLoadAnima")
    modifier_inputs = modifier["inputs"]
    assert isinstance(modifier_inputs, dict)
    assert modifier_inputs == {
        "model": [loader_id, 0],
        "clip": [loader_id, 1],
    }
    for node in built.prompt.values():
        if node["class_type"] == "CLIPTextEncode":
            node_inputs = node["inputs"]
            assert isinstance(node_inputs, dict)
            assert node_inputs["clip"] == [modifier_id, 1]
    observer_inputs = built.prompt[observer_id]["inputs"]
    assert isinstance(observer_inputs, dict)
    assert observer_inputs["model"] == [modifier_id, 0]
    assert built.modifier_snapshot_node_id == observer_id
    instrument_id = _only_node_id(
        built.prompt,
        "SimpleSyrupBenchmark.InstrumentModel",
    )
    instrument_inputs = built.prompt[instrument_id]["inputs"]
    assert isinstance(instrument_inputs, dict)
    assert instrument_inputs["model"] == [observer_id, 0]


def test_shared_builder_rejects_invalid_modifier_result_before_conditioning() -> None:
    """Require the typed boundary instead of interpreting arbitrary graph links."""

    class InvalidModifier:
        """Return a deliberately untyped graph value."""

        def add(self, *args: object, **kwargs: object) -> object:
            """Return no MODEL/CLIP contract."""

            del args, kwargs
            return object()

    with pytest.raises(TypeError, match="modifier workflow returned an invalid"):
        AnimaAttentionCouplingWorkflowBuilder(
            public_node_id="SimpleSyrup.KSamplerAttentionCoupling",
            artifact_phase="p9.7",
            conditioning=_Conditioning(),
            model_modifier_workflow=cast(
                AnimaModelModifierWorkflow,
                InvalidModifier(),
            ),
            steps=1,
        ).build(
            _Case(),
            run_id="run",
            mask_names=("left.png", "right.png"),
        )


def test_shared_builder_rejects_invalid_observer_result_before_conditioning() -> None:
    """Require evidence collaborators to return the explicit pass-through value."""

    class InvalidObserver:
        """Return a deliberately untyped observer value."""

        def add(self, *args: object, **kwargs: object) -> object:
            """Return no observed-MODEL contract."""

            del args, kwargs
            return object()

    with pytest.raises(TypeError, match="observer workflow returned an invalid"):
        AnimaAttentionCouplingWorkflowBuilder(
            public_node_id="SimpleSyrup.KSamplerAttentionCoupling",
            artifact_phase="p9.7",
            conditioning=_Conditioning(),
            model_observer_workflow=cast(
                AnimaModelObserverWorkflow,
                InvalidObserver(),
            ),
            steps=1,
        ).build(
            _Case(),
            run_id="run",
            mask_names=("left.png", "right.png"),
        )


def _only_node_id(
    prompt: dict[str, dict[str, object]],
    class_type: str,
) -> str:
    """Return the unique node ID for one graph class."""

    matches = tuple(
        node_id for node_id, node in prompt.items() if node["class_type"] == class_type
    )
    assert len(matches) == 1
    return matches[0]
