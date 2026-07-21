# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt Prompt Control and Comfy graph nodes for SEP prompt orchestration."""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Any, cast

from .prompt_control_availability import find_prompt_control_install


class PromptControlGraphAdapter:
    """Build hook-aware lazy graph fragments behind one runtime boundary."""

    def __init__(self, io: Any, graph_utils: Any, lazy_nodes: Any) -> None:
        """Store imported host graph APIs for one expansion request."""

        self.io = io
        self._graph_utils = graph_utils
        self._lazy_nodes = lazy_nodes

    @classmethod
    def load(cls, missing_message: str) -> PromptControlGraphAdapter:
        """Load Comfy and Prompt Control graph APIs or raise an actionable error."""

        try:
            io = import_module("comfy_api.latest.io")
        except ModuleNotFoundError:
            io = import_module("comfy_api.latest").io
        try:
            graph_utils = import_module("comfy_execution.graph_utils")
            lazy_nodes = cls._import_prompt_control_lazy_nodes()
        except ModuleNotFoundError as exc:
            raise RuntimeError(missing_message) from exc
        return cls(io, graph_utils, lazy_nodes)

    def schedule_global_loras(
        self,
        *,
        model: Any,
        clip: Any,
        positive_tags: str,
        negative_tags: str,
        expand: dict[str, dict[str, Any]],
    ) -> tuple[Any, Any]:
        """Preserve single-segment model/CLIP scheduling behavior."""

        positive = self._lazy_nodes.PCLazyLoraLoaderAdvanced.execute(
            model=model,
            clip=clip,
            text=positive_tags,
            apply_hooks=True,
            tags="",
            start=0.0,
            end=1.0,
            num_steps=0,
        )
        self.merge_expand(expand, positive.expand, "positive LoRA scheduling")
        negative = self._lazy_nodes.PCLazyLoraLoaderAdvanced.execute(
            model=positive.args[0],
            clip=positive.args[1],
            text=negative_tags,
            apply_hooks=True,
            tags="",
            start=0.0,
            end=1.0,
            num_steps=0,
        )
        self.merge_expand(expand, negative.expand, "negative LoRA scheduling")
        return negative.args[0], negative.args[1]

    def encode_segment(
        self,
        *,
        clip: Any,
        text: str,
        expand: dict[str, dict[str, Any]],
        label: str,
    ) -> Any:
        """Encode one segment with a previously prepared CLIP link."""

        output = self._lazy_nodes.PCLazyTextEncodeAdvanced.execute(
            clip=clip,
            text=text,
            tags="",
            start=0.0,
            end=1.0,
            num_steps=0,
        )
        self.merge_expand(expand, output.expand, f"{label} text encoding")
        return output.args[0]

    def clip_with_hooks(
        self,
        *,
        clip: Any,
        lora_tags: str,
        expand: dict[str, dict[str, Any]],
        label: str,
    ) -> Any:
        """Return a CLIP link sharing one segment's hooks across prompt sides."""

        if not lora_tags:
            return clip
        graph = self._graph_utils.GraphBuilder()
        hooks = graph.node("PCLoraHooksFromText", text=lora_tags)
        hooked_clip = graph.node(
            "SetClipHooks",
            clip=clip,
            hooks=hooks.out(0),
            apply_to_conds=True,
            schedule_clip=True,
        )
        self.merge_expand(
            expand,
            cast(dict[str, dict[str, Any]], graph.finalize()),
            f"{label} LoRA hooks",
        )
        return hooked_clip.out(0)

    def pack_conditionings(
        self,
        conditionings: list[Any],
        *,
        expand: dict[str, dict[str, Any]],
        label: str,
        always_batch: bool,
    ) -> Any:
        """Return one conditioning or a SimpleSyrup conditioning batch link."""

        if len(conditionings) == 1 and not always_batch:
            return conditionings[0]
        graph = self._graph_utils.GraphBuilder()
        current = graph.node(
            "SimpleSyrup.ConditioningBatchStart",
            conditioning=conditionings[0],
        )
        for conditioning in conditionings[1:]:
            current = graph.node(
                "SimpleSyrup.ConditioningBatchAppend",
                batch=current.out(0),
                conditioning=conditioning,
            )
        self.merge_expand(
            expand,
            cast(dict[str, dict[str, Any]], graph.finalize()),
            f"{label} batch packing",
        )
        return current.out(0)

    def merge_expand(
        self,
        target: dict[str, dict[str, Any]],
        source: object,
        operation: str,
    ) -> None:
        """Merge a graph fragment while rejecting duplicate generated ids."""

        if not source:
            return
        fragment = cast(dict[str, dict[str, Any]], source)
        overlap = set(target).intersection(fragment)
        if overlap:
            overlapping_ids = ", ".join(sorted(overlap))
            raise ValueError(
                "Prompt-Control graph expansion generated duplicate node ids "
                f"during {operation}: {overlapping_ids}."
            )
        target.update(fragment)

    @staticmethod
    def _import_prompt_control_lazy_nodes() -> Any:
        """Import Prompt Control from its normal or sibling extension path."""

        try:
            return import_module("prompt_control.nodes_lazy")
        except ModuleNotFoundError:
            availability = find_prompt_control_install()
            if availability.root_path is not None:
                root_path = str(availability.root_path)
                if root_path not in sys.path:
                    sys.path.insert(0, root_path)
            return import_module("prompt_control.nodes_lazy")
