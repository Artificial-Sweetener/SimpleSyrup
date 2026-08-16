# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build ordinary and hooked full-fidelity SDXL LoRA graph branches."""

from __future__ import annotations

from dataclasses import dataclass

from .graph import NodeReference, SdxlWorkflowGraph
from .visual_case_model import GlobalVisualAdapter, RegionalVisualAdapter


@dataclass(frozen=True, slots=True)
class LoadedGlobalVisualAdapters:
    """Expose the model and text encoders after ordinary global adapters."""

    model: NodeReference
    clip: NodeReference


class SdxlVisualLoraGraphBuilder:
    """Own full SDXL model-and-text-encoder LoRA graph construction."""

    def load_global(
        self,
        graph: SdxlWorkflowGraph,
        *,
        model: NodeReference,
        clip: NodeReference,
        adapters: tuple[GlobalVisualAdapter, ...],
    ) -> LoadedGlobalVisualAdapters:
        """Apply ordered full-image adapters to the model and both text encoders."""

        for adapter in adapters:
            loaded = graph.add(
                "LoraLoader",
                model=model,
                clip=clip,
                lora_name=adapter.lora_name,
                strength_model=adapter.strength,
                strength_clip=adapter.strength,
            )
            model = [loaded, 0]
            clip = [loaded, 1]
        return LoadedGlobalVisualAdapters(model, clip)

    def regional_hooks(
        self,
        graph: SdxlWorkflowGraph,
        adapters: tuple[RegionalVisualAdapter, ...],
    ) -> tuple[NodeReference | None, tuple[str, ...]]:
        """Create and combine ordered full-fidelity regional LoRA hooks."""

        created = tuple(self._regional_hook(graph, adapter) for adapter in adapters)
        if not created:
            return None, ()
        identities = tuple(adapter.lora_name for adapter in adapters)
        if len(set(identities)) != len(identities):
            raise ValueError("One region cannot repeat the same adapter identity.")
        hooks = created[0]
        for added in created[1:]:
            combined = graph.add("CombineHooks2", hooks_A=hooks, hooks_B=added)
            hooks = [combined, 0]
        return hooks, identities

    @staticmethod
    def _regional_hook(
        graph: SdxlWorkflowGraph,
        adapter: RegionalVisualAdapter,
    ) -> NodeReference:
        """Create one independently scheduled model-and-CLIP LoRA hook."""

        hook = graph.add(
            "CreateHookLora",
            lora_name=adapter.lora_name,
            strength_model=adapter.model_strength,
            strength_clip=adapter.clip_strength,
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
