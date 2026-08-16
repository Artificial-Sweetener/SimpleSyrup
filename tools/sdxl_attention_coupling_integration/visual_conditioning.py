# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build native dual-encoder SDXL conditioning with regional LoRA hooks."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .graph import NodeReference, SdxlWorkflowGraph
from .matrix import (
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)
from .visual_case_model import RegionalVisualAdapter, SdxlVisualCase
from .visual_lora_graph import SdxlVisualLoraGraphBuilder
from .visual_prompt_schedule import schedule_visual_regional_prompt


@dataclass(frozen=True, slots=True)
class BuiltSdxlVisualConditioning:
    """Expose the model and global-first conditioning batches for sampling."""

    model: NodeReference
    positive: NodeReference
    negative: NodeReference


class SdxlVisualConditioningBuilder:
    """Own ordinary global LoRAs and regional SDXL HookGroup conditioning."""

    def build(
        self,
        graph: SdxlWorkflowGraph,
        *,
        case: SdxlVisualCase,
        model: NodeReference,
        clip: NodeReference,
    ) -> BuiltSdxlVisualConditioning:
        """Build one native G/L conditioning batch for the declared case."""

        loaded = SdxlVisualLoraGraphBuilder().load_global(
            graph,
            model=model,
            clip=clip,
            adapters=case.global_adapters,
        )
        model = loaded.model
        clip = loaded.clip
        shared_positive_g = self._with_optional(
            case.base_positive_g,
            case.global_style_g,
        )
        shared_positive_l = self._with_optional(
            case.base_positive_l,
            case.global_style_l,
        )
        base_positive = self._encode(
            graph,
            clip=clip,
            text_g=shared_positive_g,
            text_l=shared_positive_l,
        )
        base_negative = self._encode(
            graph,
            clip=clip,
            text_g=case.base_negative_g,
            text_l=case.base_negative_l,
        )
        left_has_regional_negative = self._has_regional_negative(
            case.left_negative_g,
            case.left_negative_l,
        )
        right_has_regional_negative = self._has_regional_negative(
            case.right_negative_g,
            case.right_negative_l,
        )
        right_requires_negative = right_has_regional_negative or bool(
            case.right_adapters
        )
        left_requires_negative = (
            left_has_regional_negative
            or bool(case.left_adapters)
            or right_requires_negative
        )
        left_positive, left_negative = self._regional_conditionings(
            graph,
            clip=clip,
            positive_g=self._with_optional(shared_positive_g, case.left_g),
            positive_l=self._with_optional(shared_positive_l, case.left_l),
            negative_g=self._with_optional(
                case.base_negative_g,
                case.left_negative_g,
            ),
            negative_l=self._with_optional(
                case.base_negative_l,
                case.left_negative_l,
            ),
            include_regional_negative=left_requires_negative,
            adapters=case.left_adapters,
            start_percent=case.regional_prompt_start_percent,
        )
        right_positive, right_negative = self._regional_conditionings(
            graph,
            clip=clip,
            positive_g=self._with_optional(shared_positive_g, case.right_g),
            positive_l=self._with_optional(shared_positive_l, case.right_l),
            negative_g=self._with_optional(
                case.base_negative_g,
                case.right_negative_g,
            ),
            negative_l=self._with_optional(
                case.base_negative_l,
                case.right_negative_l,
            ),
            include_regional_negative=right_requires_negative,
            adapters=case.right_adapters,
            start_percent=case.regional_prompt_start_percent,
        )
        negative_entries = tuple(
            conditioning
            for conditioning in (base_negative, left_negative, right_negative)
            if conditioning is not None
        )
        return BuiltSdxlVisualConditioning(
            model,
            self._pack(graph, (base_positive, left_positive, right_positive)),
            self._pack(graph, negative_entries),
        )

    @staticmethod
    def _with_optional(prompt: str, addition: str) -> str:
        """Append one non-empty externally supplied prompt fragment."""

        return f"{prompt}, {addition}" if addition else prompt

    @staticmethod
    def _has_regional_negative(negative_g: str, negative_l: str) -> bool:
        """Require regional G and L negatives to be authored as one pair."""

        has_g = bool(negative_g.strip())
        has_l = bool(negative_l.strip())
        if has_g != has_l:
            raise ValueError("Regional negative G and L prompts must be paired.")
        return has_g

    def _regional_conditionings(
        self,
        graph: SdxlWorkflowGraph,
        *,
        clip: NodeReference,
        positive_g: str,
        positive_l: str,
        negative_g: str,
        negative_l: str,
        include_regional_negative: bool,
        adapters: tuple[RegionalVisualAdapter, ...],
        start_percent: float,
    ) -> tuple[NodeReference, NodeReference | None]:
        """Encode paired hooked CFG entries or one prompt-only positive."""

        hooks = self._hooks(graph, adapters)
        regional_clip = clip
        if hooks is not None:
            prepared = graph.add(
                "SimpleSyrup.PrepareRegionalLoraHooks",
                clip=clip,
                hooks=hooks,
            )
            regional_clip = [prepared, 0]
            hooks = [prepared, 1]
        positive = self._encode(
            graph,
            clip=regional_clip,
            text_g=positive_g,
            text_l=positive_l,
        )
        if hooks is not None:
            positive = self._attach(graph, positive, hooks)
        positive = schedule_visual_regional_prompt(
            graph,
            positive,
            start_percent=start_percent,
        )
        negative: NodeReference | None = None
        if include_regional_negative:
            negative = self._encode(
                graph,
                clip=regional_clip,
                text_g=negative_g,
                text_l=negative_l,
            )
            if hooks is not None:
                negative = self._attach(graph, negative, hooks)
            negative = schedule_visual_regional_prompt(
                graph,
                negative,
                start_percent=start_percent,
            )
        return positive, negative

    def _hooks(
        self,
        graph: SdxlWorkflowGraph,
        adapters: tuple[RegionalVisualAdapter, ...],
    ) -> NodeReference | None:
        """Create, schedule, combine, and label one region's ordered hooks."""

        hooks, identities = SdxlVisualLoraGraphBuilder().regional_hooks(
            graph,
            adapters,
        )
        if hooks is None:
            return None
        labeled = graph.add(
            "SimpleSyrup.LabelRegionalLoraHooks",
            hooks=hooks,
            adapter_identities_json=json.dumps(identities, separators=(",", ":")),
        )
        return [labeled, 0]

    @staticmethod
    def _encode(
        graph: SdxlWorkflowGraph,
        *,
        clip: NodeReference,
        text_g: str,
        text_l: str,
    ) -> NodeReference:
        """Encode native SDXL G/L text with explicit microconditioning."""

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

    @staticmethod
    def _attach(
        graph: SdxlWorkflowGraph,
        conditioning: NodeReference,
        hooks: NodeReference,
    ) -> NodeReference:
        """Attach one region's hooks through Comfy's public conditioning node."""

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
        graph: SdxlWorkflowGraph,
        entries: tuple[NodeReference, ...],
    ) -> NodeReference:
        """Pack the unmasked base then the two ordered regional entries."""

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
