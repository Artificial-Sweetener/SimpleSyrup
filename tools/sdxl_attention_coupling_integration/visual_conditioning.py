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
from .visual_cases import (
    BASE_NEGATIVE_G,
    BASE_NEGATIVE_L,
    BASE_POSITIVE_G,
    BASE_POSITIVE_L,
    RegionalVisualAdapter,
    SdxlVisualCase,
)


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

        for adapter in case.global_adapters:
            loaded = graph.add(
                "LoraLoaderModelOnly",
                model=model,
                lora_name=adapter.lora_name,
                strength_model=adapter.strength,
            )
            model = [loaded, 0]
        base_positive = self._encode(
            graph,
            clip=clip,
            text_g=BASE_POSITIVE_G,
            text_l=BASE_POSITIVE_L,
        )
        base_negative = self._encode(
            graph,
            clip=clip,
            text_g=BASE_NEGATIVE_G,
            text_l=BASE_NEGATIVE_L,
        )
        left_positive, left_negative = self._regional_pair(
            graph,
            clip=clip,
            positive_g=case.left_g,
            positive_l=case.left_l,
            adapters=case.left_adapters,
        )
        right_positive, right_negative = self._regional_pair(
            graph,
            clip=clip,
            positive_g=case.right_g,
            positive_l=case.right_l,
            adapters=case.right_adapters,
        )
        return BuiltSdxlVisualConditioning(
            model,
            self._pack(graph, (base_positive, left_positive, right_positive)),
            self._pack(graph, (base_negative, left_negative, right_negative)),
        )

    def _regional_pair(
        self,
        graph: SdxlWorkflowGraph,
        *,
        clip: NodeReference,
        positive_g: str,
        positive_l: str,
        adapters: tuple[RegionalVisualAdapter, ...],
    ) -> tuple[NodeReference, NodeReference]:
        """Encode and hook one region's positive and negative SDXL entries."""

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
        negative = self._encode(
            graph,
            clip=regional_clip,
            text_g=BASE_NEGATIVE_G,
            text_l=BASE_NEGATIVE_L,
        )
        if hooks is not None:
            positive = self._attach(graph, positive, hooks)
            negative = self._attach(graph, negative, hooks)
        return positive, negative

    def _hooks(
        self,
        graph: SdxlWorkflowGraph,
        adapters: tuple[RegionalVisualAdapter, ...],
    ) -> NodeReference | None:
        """Create, schedule, combine, and label one region's ordered hooks."""

        created = tuple(self._hook(graph, adapter) for adapter in adapters)
        if not created:
            return None
        hooks = created[0]
        for added in created[1:]:
            combined = graph.add("CombineHooks2", hooks_A=hooks, hooks_B=added)
            hooks = [combined, 0]
        identities = [adapter.lora_name for adapter in adapters]
        if len(set(identities)) != len(identities):
            raise ValueError("One region cannot repeat the same adapter identity.")
        labeled = graph.add(
            "SimpleSyrup.LabelRegionalLoraHooks",
            hooks=hooks,
            adapter_identities_json=json.dumps(identities, separators=(",", ":")),
        )
        return [labeled, 0]

    @staticmethod
    def _hook(
        graph: SdxlWorkflowGraph,
        adapter: RegionalVisualAdapter,
    ) -> NodeReference:
        """Create one independently scheduled model-only LoRA hook."""

        hook = graph.add(
            "CreateHookLoraModelOnly",
            lora_name=adapter.lora_name,
            strength_model=adapter.strength,
        )
        if adapter.schedule == ((0.0, 1.0),):
            return [hook, 0]
        previous: NodeReference | None = None
        for start_percent, strength_mult in adapter.schedule:
            inputs: dict[str, object] = {
                "strength_mult": strength_mult,
                "start_percent": start_percent,
            }
            if previous is not None:
                inputs["prev_hook_kf"] = previous
            keyframe = graph.add("CreateHookKeyframe", **inputs)
            previous = [keyframe, 0]
        if previous is None:
            raise ValueError("Regional adapter schedule cannot be empty.")
        scheduled = graph.add("SetHookKeyframes", hooks=[hook, 0], hook_kf=previous)
        return [scheduled, 0]

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
