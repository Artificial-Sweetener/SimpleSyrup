# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own exact Prompt Control text construction shared by managed evidence."""

from __future__ import annotations

from dataclasses import dataclass

from tools.anima_workflow_graph import AnimaWorkflowGraph, NodeReference

from .cases import PromptControlCase

ADJACENT_TEXT = "[red subject:blue subject:0.5]"


@dataclass(frozen=True, slots=True)
class PromptControlExpansionInputs:
    """Retain exact installed-lazy-node expansion probe inputs."""

    text: str
    capture_text_expansion: bool


def expansion_inputs(case: PromptControlCase) -> PromptControlExpansionInputs:
    """Return expansion inputs without evaluating Prompt Control syntax."""

    if case.text_construction == "adjacent":
        return PromptControlExpansionInputs(ADJACENT_TEXT, True)
    return PromptControlExpansionInputs(
        case.positive_text,
        case.text_construction == "lazy",
    )


class PromptControlTextConditioningBuilder:
    """Build one exact public text-conditioning shape from an immutable case."""

    def add(
        self,
        graph: AnimaWorkflowGraph,
        case: PromptControlCase,
        *,
        clip: NodeReference,
    ) -> NodeReference:
        """Return the exact lazy, adjacent, overlap, or inactive graph link."""

        if case.text_construction == "lazy":
            node = graph.add("PCLazyTextEncode", clip=clip, text=case.positive_text)
            return [node, 0]
        if case.text_construction == "adjacent":
            node = graph.add("PCLazyTextEncode", clip=clip, text=ADJACENT_TEXT)
            return [node, 0]
        if case.text_construction == "overlapping":
            first = graph.add(
                "PCTextEncodeWithRange",
                clip=clip,
                text="first subject",
                start=0.0,
                end=0.75,
            )
            second = graph.add(
                "PCTextEncodeWithRange",
                clip=clip,
                text="second subject",
                start=0.25,
                end=1.0,
            )
            combined = graph.add(
                "ConditioningCombine",
                conditioning_1=[first, 0],
                conditioning_2=[second, 0],
            )
            return [combined, 0]
        node = graph.add(
            "PCTextEncodeWithRange",
            clip=clip,
            text=case.positive_text,
            start=0.5,
            end=0.5,
        )
        return [node, 0]


PROMPT_CONTROL_TEXT_CONDITIONING_BUILDER = PromptControlTextConditioningBuilder()
