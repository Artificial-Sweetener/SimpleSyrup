# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Runtime graph expansion for Prompt-Control scheduling and prompt encoding."""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Any, cast

from ..domain.prompt_control_prompt import (
    PreparedPromptSide,
    apply_encode_style,
    prepare_prompt_side,
)
from .prompt_control_availability import find_prompt_control_install

PROMPT_CONTROL_MISSING_MESSAGE = (
    "Schedule & Encode Prompts requires comfyui-prompt-control. "
    "Install Prompt Control or remove this node from the workflow."
)
PROMPT_BATCH_SEPARATOR = "[SEP]"


class PromptControlScheduleEncodeGraphBuilder:
    """Build lazy Prompt-Control graphs for LoRA scheduling and prompt encoding."""

    def build(
        self,
        model: Any,
        clip: Any,
        positive_prompt: str,
        negative_prompt: str,
        encode_style: str = "",
    ) -> Any:
        """Return an io.NodeOutput for scheduled model and encoded prompts."""

        io, graph_utils, lazy_nodes = self._prompt_control_dependencies()
        positive_side = prepare_prompt_side(positive_prompt, PROMPT_BATCH_SEPARATOR)
        negative_side = prepare_prompt_side(negative_prompt, PROMPT_BATCH_SEPARATOR)

        expand: dict[str, dict[str, Any]] = {}
        positive_lora = lazy_nodes.PCLazyLoraLoaderAdvanced.execute(
            model=model,
            clip=clip,
            text=positive_side.lora_tags,
            apply_hooks=True,
            tags="",
            start=0.0,
            end=1.0,
            num_steps=0,
        )
        self._merge_expand(expand, positive_lora.expand, "positive LoRA scheduling")

        negative_lora = lazy_nodes.PCLazyLoraLoaderAdvanced.execute(
            model=positive_lora.args[0],
            clip=positive_lora.args[1],
            text=negative_side.lora_tags,
            apply_hooks=True,
            tags="",
            start=0.0,
            end=1.0,
            num_steps=0,
        )
        self._merge_expand(expand, negative_lora.expand, "negative LoRA scheduling")

        scheduled_model = negative_lora.args[0]
        scheduled_clip = negative_lora.args[1]
        positive_conditioning = self._encode_side(
            side=positive_side,
            clip=scheduled_clip,
            encode_style=encode_style,
            graph_utils=graph_utils,
            lazy_text_encoder=lazy_nodes.PCLazyTextEncodeAdvanced,
            expand=expand,
            label="positive prompt encoding",
        )
        negative_conditioning = self._encode_side(
            side=negative_side,
            clip=scheduled_clip,
            encode_style=encode_style,
            graph_utils=graph_utils,
            lazy_text_encoder=lazy_nodes.PCLazyTextEncodeAdvanced,
            expand=expand,
            label="negative prompt encoding",
        )

        return io.NodeOutput(
            scheduled_model,
            positive_conditioning,
            negative_conditioning,
            expand=expand,
        )

    def _encode_side(
        self,
        *,
        side: PreparedPromptSide,
        clip: Any,
        encode_style: str,
        graph_utils: Any,
        lazy_text_encoder: Any,
        expand: dict[str, dict[str, Any]],
        label: str,
    ) -> Any:
        """Encode one prompt side and return conditioning or conditioning batch."""

        conditioning_outputs: list[Any] = []
        for index, chunk in enumerate(side.chunks):
            text = apply_encode_style(encode_style, chunk.text)
            node_output = lazy_text_encoder.execute(
                clip=clip,
                text=text,
                tags="",
                start=0.0,
                end=1.0,
                num_steps=0,
            )
            self._merge_expand(
                expand,
                node_output.expand,
                f"{label} chunk {index}",
            )
            conditioning_outputs.append(node_output.args[0])

        if len(conditioning_outputs) == 1:
            return conditioning_outputs[0]

        pack_graph = graph_utils.GraphBuilder()
        current = pack_graph.node(
            "SimpleSyrup.ConditioningBatchStart",
            conditioning=conditioning_outputs[0],
        )
        for conditioning in conditioning_outputs[1:]:
            current = pack_graph.node(
                "SimpleSyrup.ConditioningBatchAppend",
                batch=current.out(0),
                conditioning=conditioning,
            )
        self._merge_expand(
            expand,
            cast(dict[str, dict[str, Any]], pack_graph.finalize()),
            f"{label} batch packing",
        )
        return current.out(0)

    def _prompt_control_dependencies(self) -> tuple[Any, Any, Any]:
        """Import Prompt-Control and Comfy graph helpers on demand."""

        try:
            io = import_module("comfy_api.latest.io")
        except ModuleNotFoundError:
            comfy_api = import_module("comfy_api.latest")
            io = comfy_api.io
        try:
            graph_utils = import_module("comfy_execution.graph_utils")
            lazy_nodes = self._import_prompt_control_lazy_nodes()
        except ModuleNotFoundError as exc:
            raise RuntimeError(PROMPT_CONTROL_MISSING_MESSAGE) from exc
        return io, graph_utils, lazy_nodes

    def _import_prompt_control_lazy_nodes(self) -> Any:
        """Import Prompt-Control lazy nodes from installed or sibling paths."""

        try:
            return import_module("prompt_control.nodes_lazy")
        except ModuleNotFoundError:
            availability = find_prompt_control_install()
            if availability.root_path is not None:
                root_path = str(availability.root_path)
                if root_path not in sys.path:
                    sys.path.insert(0, root_path)
            return import_module("prompt_control.nodes_lazy")

    def _merge_expand(
        self,
        target: dict[str, dict[str, Any]],
        source: object,
        operation: str,
    ) -> None:
        """Merge a lazy expand graph and reject duplicate generated node ids."""

        if not source:
            return
        expand = cast(dict[str, dict[str, Any]], source)
        overlap = set(target).intersection(expand)
        if overlap:
            overlapping_ids = ", ".join(sorted(overlap))
            raise ValueError(
                f"Prompt-Control graph expansion generated duplicate node ids "
                f"during {operation}: {overlapping_ids}."
            )
        target.update(expand)
