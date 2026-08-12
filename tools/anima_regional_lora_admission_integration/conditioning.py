"""Build native regional-LoRA admission conditioning and HookGroup graphs."""

from __future__ import annotations

import json

from tools.anima_attention_coupling_conditioning import BuiltAnimaConditioning
from tools.anima_workflow_graph import AnimaWorkflowGraph, NodeReference

from .graph_contract import (
    BASE_PROMPT,
    NEGATIVE_PROMPT,
    REGION_ONE_PROMPT,
    REGION_ZERO_PROMPT,
    RegionalLoraAdmissionGraphCase,
)


class RegionalLoraAdmissionConditioningWorkflow:
    """Build one base-plus-two-region batch with one declared hook source."""

    def __init__(self, case: RegionalLoraAdmissionGraphCase) -> None:
        """Retain one immutable matrix case."""

        if not isinstance(case, RegionalLoraAdmissionGraphCase):
            raise TypeError("Regional LoRA conditioning requires a graph case.")
        self._case = case

    def add(
        self,
        graph: AnimaWorkflowGraph,
        *,
        model: NodeReference,
        clip: NodeReference,
    ) -> BuiltAnimaConditioning:
        """Return the unchanged model plus native conditioning-batch links."""

        positive = (
            self._encode(graph, clip, BASE_PROMPT),
            self._encode(graph, clip, REGION_ZERO_PROMPT),
            self._encode(graph, clip, REGION_ONE_PROMPT),
        )
        negative: tuple[NodeReference, NodeReference, NodeReference] = (
            self._encode(graph, clip, NEGATIVE_PROMPT),
            self._encode(graph, clip, NEGATIVE_PROMPT),
            self._encode(graph, clip, NEGATIVE_PROMPT),
        )
        hooks = self._hooks(graph)
        if hooks is None:
            raise ValueError("Regional LoRA admission cases must declare hooks.")
        positive = (
            positive[0],
            self._attach_hooks(graph, positive[1], hooks),
            positive[2],
        )
        return BuiltAnimaConditioning(
            model,
            self._pack(graph, positive),
            self._pack(graph, negative),
        )

    def _hooks(self, graph: AnimaWorkflowGraph) -> NodeReference | None:
        """Construct, combine, and label the case's ordered installed hooks."""

        hooks: NodeReference | None = None
        for adapter in self._case.public_loras:
            public_hook = graph.add(
                "CreateHookLoraModelOnly",
                lora_name=adapter.lora_name,
                strength_model=adapter.strength_model,
            )
            hooks = self._combine_hooks(graph, hooks, [public_hook, 0])
        if self._case.fixture is not None:
            fixture = graph.add(
                "SimpleSyrupBenchmark.CreateRegionalHookFixture",
                fixture=self._case.fixture,
            )
            hooks = self._combine_hooks(graph, hooks, [fixture, 0])
        if hooks is not None and self._case.adapter_identities:
            labeled = graph.add(
                "SimpleSyrup.LabelRegionalLoraHooks",
                hooks=hooks,
                adapter_identities_json=json.dumps(
                    self._case.adapter_identities,
                    separators=(",", ":"),
                ),
            )
            hooks = [labeled, 0]
        return hooks

    @staticmethod
    def _combine_hooks(
        graph: AnimaWorkflowGraph,
        existing: NodeReference | None,
        added: NodeReference,
    ) -> NodeReference:
        """Append one public HookGroup while preserving declared order."""

        if existing is None:
            return added
        combined = graph.add(
            "CombineHooks2",
            hooks_A=existing,
            hooks_B=added,
        )
        return [combined, 0]

    @staticmethod
    def _encode(
        graph: AnimaWorkflowGraph,
        clip: NodeReference,
        text: str,
    ) -> NodeReference:
        """Encode one ordinary full-range conditioning."""

        node = graph.add("CLIPTextEncode", clip=clip, text=text)
        return [node, 0]

    @staticmethod
    def _attach_hooks(
        graph: AnimaWorkflowGraph,
        conditioning: NodeReference,
        hooks: NodeReference,
    ) -> NodeReference:
        """Attach the regional hook source through Comfy's public node."""

        node = graph.add(
            "ConditioningSetProperties",
            cond_NEW=conditioning,
            hooks=hooks,
            strength=1.0,
            set_cond_area="default",
        )
        return [node, 0]

    @staticmethod
    def _pack(
        graph: AnimaWorkflowGraph,
        conditionings: tuple[NodeReference, NodeReference, NodeReference],
    ) -> NodeReference:
        """Pack the global-first three-entry public conditioning batch."""

        current = graph.add(
            "SimpleSyrup.ConditioningBatchStart",
            conditioning=conditionings[0],
        )
        for conditioning in conditionings[1:]:
            current = graph.add(
                "SimpleSyrup.ConditioningBatchAppend",
                batch=[current, 0],
                conditioning=conditioning,
            )
        return [current, 0]
