# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build exact P9.1 regional Prompt Control conditioning graphs."""

from __future__ import annotations

import json

from tools.anima_attention_coupling_conditioning import (
    AnimaConditioningEvidence,
    BuiltAnimaConditioning,
)
from tools.anima_workflow_graph import AnimaWorkflowGraph, NodeReference
from tools.prompt_control_characterization.conditioning import (
    PROMPT_CONTROL_TEXT_CONDITIONING_BUILDER,
    expansion_inputs,
)

from .matrix import (
    BASE_PROMPT,
    NEGATIVE_PROMPT,
    REGION_ONE_PROMPT,
    REGION_ZERO_NEGATIVE_PROMPT,
    PromptControlAttentionCase,
)


class PromptControlAttentionConditioningWorkflow:
    """Build one base-plus-two-region batch using public Prompt Control hooks."""

    def __init__(self, case: PromptControlAttentionCase) -> None:
        """Retain one immutable matrix case."""

        if not isinstance(case, PromptControlAttentionCase):
            raise TypeError("P9.1 conditioning requires a matrix case.")
        self._case = case

    def add(
        self,
        graph: AnimaWorkflowGraph,
        *,
        model: NodeReference,
        clip: NodeReference,
    ) -> BuiltAnimaConditioning:
        """Return unchanged model and public conditioning-batch links."""

        case = self._case.characterization
        raw_clip = clip
        expansion_spec = expansion_inputs(case)
        expansion = graph.add(
            "SimpleSyrupBenchmark.SnapshotPromptControlExpansion",
            case_id=case.case_id,
            text=expansion_spec.text,
            capture_text_expansion=expansion_spec.capture_text_expansion,
            lora_text=case.lora_text,
        )
        regional_clip = raw_clip
        hooks: NodeReference | None = None
        if self._case.has_regional_hooks:
            scheduled = graph.add(
                "PCLazyLoraLoaderAdvanced",
                text=case.lora_text,
                apply_hooks=True,
                tags="",
                start=0.0,
                end=1.0,
                num_steps=0,
            )
            labeled = graph.add(
                "SimpleSyrup.LabelRegionalLoraHooks",
                hooks=[scheduled, 2],
                adapter_identities_json=json.dumps(self._case.adapter_identities),
            )
            prepared = graph.add(
                "SimpleSyrup.PrepareRegionalLoraHooks",
                clip=raw_clip,
                hooks=[labeled, 0],
            )
            regional_clip = [prepared, 0]
            hooks = [prepared, 1]

        regional_positive = PROMPT_CONTROL_TEXT_CONDITIONING_BUILDER.add(
            graph,
            case,
            clip=regional_clip,
        )
        regional_negative = self._encode_characterized_negative(
            graph,
            regional_clip,
        )
        if hooks is not None:
            regional_positive = self._attach_hooks(graph, regional_positive, hooks)
            regional_negative = self._attach_hooks(graph, regional_negative, hooks)
        snapshot_inputs: dict[str, object] = {
            "positive": regional_positive,
            "negative": regional_negative,
            "case_id": case.case_id,
            "adapter_identities_json": json.dumps(self._case.adapter_identities),
        }
        if hooks is not None:
            snapshot_inputs["hooks"] = hooks
        snapshot = graph.add(
            "SimpleSyrupBenchmark.SnapshotPromptControl",
            **snapshot_inputs,
        )
        regional_positive = [snapshot, 0]
        regional_negative = [snapshot, 1]

        base_positive = self._encode_range(graph, raw_clip, BASE_PROMPT)
        base_negative = self._encode_range(graph, raw_clip, NEGATIVE_PROMPT)
        second_positive = self._encode_range(graph, raw_clip, REGION_ONE_PROMPT)
        second_negative = self._encode_range(graph, raw_clip, NEGATIVE_PROMPT)
        if hooks is not None:
            regional_positive = self._attach_global_companion(
                graph,
                regional_positive,
                self._hooked_companion(graph, regional_clip, hooks, BASE_PROMPT),
            )
            regional_negative = self._attach_global_companion(
                graph,
                regional_negative,
                self._hooked_companion(
                    graph,
                    regional_clip,
                    hooks,
                    NEGATIVE_PROMPT,
                ),
            )
        positive = self._pack(
            graph,
            (base_positive, regional_positive, second_positive),
        )
        negative = self._pack(
            graph,
            (base_negative, regional_negative, second_negative),
        )
        return BuiltAnimaConditioning(
            model,
            positive,
            negative,
            AnimaConditioningEvidence(expansion, snapshot),
        )

    @staticmethod
    def _encode_range(
        graph: AnimaWorkflowGraph,
        clip: NodeReference,
        text: str,
    ) -> NodeReference:
        """Encode one full-range public Prompt Control conditioning."""

        return PromptControlAttentionConditioningWorkflow._encode_range_between(
            graph,
            clip,
            text,
            start=0.0,
            end=1.0,
        )

    def _encode_characterized_negative(
        self,
        graph: AnimaWorkflowGraph,
        clip: NodeReference,
    ) -> NodeReference:
        """Encode exact P0.8 negative intervals without schedule interpretation."""

        entries = tuple(
            self._encode_range_between(
                graph,
                clip,
                REGION_ZERO_NEGATIVE_PROMPT,
                start=expected.start,
                end=expected.end,
            )
            for expected in self._case.characterization.expected_negative
        )
        current = entries[0]
        for entry in entries[1:]:
            combined = graph.add(
                "ConditioningCombine",
                conditioning_1=current,
                conditioning_2=entry,
            )
            current = [combined, 0]
        return current

    @staticmethod
    def _encode_range_between(
        graph: AnimaWorkflowGraph,
        clip: NodeReference,
        text: str,
        *,
        start: float,
        end: float,
    ) -> NodeReference:
        """Encode one explicit characterized conditioning range."""

        node = graph.add(
            "PCTextEncodeWithRange",
            clip=clip,
            text=text,
            start=start,
            end=end,
        )
        return [node, 0]

    @staticmethod
    def _attach_hooks(
        graph: AnimaWorkflowGraph,
        conditioning: NodeReference,
        hooks: NodeReference,
    ) -> NodeReference:
        """Attach prepared regional model hooks to one conditioning."""

        node = graph.add(
            "ConditioningSetProperties",
            cond_NEW=conditioning,
            hooks=hooks,
            strength=1.0,
            set_cond_area="default",
        )
        return [node, 0]

    @classmethod
    def _hooked_companion(
        cls,
        graph: AnimaWorkflowGraph,
        clip: NodeReference,
        hooks: NodeReference,
        text: str,
    ) -> NodeReference:
        """Encode and attach hooks to one global companion conditioning."""

        return cls._attach_hooks(graph, cls._encode_range(graph, clip, text), hooks)

    @staticmethod
    def _attach_global_companion(
        graph: AnimaWorkflowGraph,
        conditioning: NodeReference,
        global_conditioning: NodeReference,
    ) -> NodeReference:
        """Attach one hook-aware global share to a regional conditioning."""

        node = graph.add(
            "SimpleSyrup.AttachRegionalGlobalConditioning",
            conditioning=conditioning,
            global_conditioning=global_conditioning,
        )
        return [node, 0]

    @staticmethod
    def _pack(
        graph: AnimaWorkflowGraph,
        conditionings: tuple[NodeReference, ...],
    ) -> NodeReference:
        """Pack base and two ordered regions through the public batch nodes."""

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
