# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Runtime graph expansion for Prompt Control conditioning batches."""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Any, cast

from ..domain.conditioning_batch import split_prompt_batch
from .prompt_control_availability import find_prompt_control_install

PROMPT_CONTROL_MISSING_MESSAGE = (
    "Encode Prompt Batch w/ Prompt Control requires comfyui-prompt-control. "
    "Install Prompt Control or use Encode Prompt Batch."
)


class PromptControlBatchGraphBuilder:
    """Build lazy Prompt Control graphs that return conditioning batches."""

    def build(
        self,
        clip: Any,
        positive_prompt: str,
        negative_prompt: str,
        separator: str,
    ) -> Any:
        """Return an io.NodeOutput with positive and negative batch links."""

        io, graph_utils, lazy_node = self._prompt_control_dependencies()
        positive_chunks = split_prompt_batch(positive_prompt, separator)
        negative_chunks = split_prompt_batch(negative_prompt, separator)

        expand: dict[str, dict[str, Any]] = {}
        positive_output, positive_expand = self._encode_chunks(
            chunks=positive_chunks,
            clip=clip,
            graph_utils=graph_utils,
            lazy_node=lazy_node,
        )
        negative_output, negative_expand = self._encode_chunks(
            chunks=negative_chunks,
            clip=clip,
            graph_utils=graph_utils,
            lazy_node=lazy_node,
        )
        expand.update(positive_expand)
        expand.update(negative_expand)
        return io.NodeOutput(positive_output, negative_output, expand=expand)

    def _encode_chunks(
        self,
        chunks: tuple[str, ...],
        clip: Any,
        graph_utils: Any,
        lazy_node: Any,
    ) -> tuple[list[Any], dict[str, dict[str, Any]]]:
        """Encode chunks with Prompt Control and pack the resulting links."""

        expand: dict[str, dict[str, Any]] = {}
        conditioning_outputs: list[Any] = []
        for chunk in chunks:
            node_output = lazy_node.execute(
                clip=clip,
                text=chunk,
                tags="",
                start=0.0,
                end=1.0,
                num_steps=0,
            )
            node_expand = cast(dict[str, dict[str, Any]], node_output.expand or {})
            overlap = set(expand).intersection(node_expand)
            if overlap:
                overlapping_ids = ", ".join(sorted(overlap))
                raise ValueError(
                    "Prompt Control generated duplicate graph node ids: "
                    f"{overlapping_ids}."
                )
            expand.update(node_expand)
            conditioning_outputs.append(node_output.args[0])

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
        pack_expand = cast(dict[str, dict[str, Any]], pack_graph.finalize())
        overlap = set(expand).intersection(pack_expand)
        if overlap:
            overlapping_ids = ", ".join(sorted(overlap))
            raise ValueError(
                f"SimpleSyrup generated duplicate graph node ids: {overlapping_ids}."
            )
        expand.update(pack_expand)
        return current.out(0), expand

    def _prompt_control_dependencies(self) -> tuple[Any, Any, Any]:
        """Import Prompt Control and Comfy v3 dependencies on demand."""

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
        return io, graph_utils, lazy_nodes.PCLazyTextEncodeAdvanced

    def _import_prompt_control_lazy_nodes(self) -> Any:
        """Import Prompt Control lazy nodes from normal or sibling extension paths."""

        try:
            return import_module("prompt_control.nodes_lazy")
        except ModuleNotFoundError:
            availability = find_prompt_control_install()
            if availability.root_path is not None:
                root_path = str(availability.root_path)
                if root_path not in sys.path:
                    sys.path.insert(0, root_path)
            return import_module("prompt_control.nodes_lazy")
