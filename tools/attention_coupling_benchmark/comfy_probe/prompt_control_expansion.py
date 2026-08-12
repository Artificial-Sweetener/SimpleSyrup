"""Capture canonical graphs returned by Prompt Control public lazy nodes."""

from __future__ import annotations

import json
from importlib import import_module
from typing import TYPE_CHECKING, Any

_comfy_api: Any = None
if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for the benchmark-only expansion node."""

        pass

else:
    _comfy_api = import_module("comfy_api.latest")
    _ComfyNodeBase = _comfy_api.io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else _comfy_api.io


class SnapshotPromptControlExpansionV3(_ComfyNodeBase):
    """Invoke installed lazy nodes and return their canonical expansion graphs."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare source text inputs and one JSON evidence output."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.SnapshotPromptControlExpansion",
            display_name="Benchmark Snapshot Prompt Control Expansion",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.String.Input("case_id"),
                _comfy_io.String.Input("text"),
                _comfy_io.Boolean.Input("capture_text_expansion"),
                _comfy_io.String.Input("lora_text"),
            ],
            outputs=[_comfy_io.String.Output("expansion_json")],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        case_id: str,
        text: str,
        capture_text_expansion: bool,
        lora_text: str,
    ) -> Any:
        """Call the pinned implementation without duplicating its parser."""

        module = import_module(
            "custom_nodes.comfyui-prompt-control.prompt_control.nodes_lazy"
        )
        text_graph: dict[str, object] = {}
        if capture_text_expansion:
            text_output = module.PCLazyTextEncode.execute(["clip-source", 0], text)
            text_graph = canonicalize_expansion(text_output.expand)
        lora_graph: dict[str, object] = {}
        if lora_text:
            lora_output = module.PCLazyLoraLoaderAdvanced.execute(
                model=["model-source", 0],
                clip=["clip-source", 0],
                text=lora_text,
                apply_hooks=True,
                tags="",
                start=0.0,
                end=1.0,
                num_steps=0,
            )
            lora_graph = canonicalize_expansion(lora_output.expand)
        snapshot = {
            "case_id": case_id,
            "text_expansion": text_graph,
            "lora_expansion": lora_graph,
        }
        encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        return _comfy_io.NodeOutput(
            encoded,
            ui={"prompt_control_expansion": [snapshot]},
        )


def canonicalize_expansion(value: object) -> dict[str, object]:
    """Rename expansion node IDs and links in returned creation order."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError("Prompt Control expansion must be a graph object.")
    mapping = {node_id: f"node-{index}" for index, node_id in enumerate(value, 1)}
    result: dict[str, object] = {}
    for node_id, raw_node in value.items():
        if not isinstance(raw_node, dict) or not all(
            isinstance(key, str) for key in raw_node
        ):
            raise TypeError("Prompt Control expansion nodes must be objects.")
        result[mapping[node_id]] = _replace_links(raw_node, mapping)
    return result


def _replace_links(value: object, mapping: dict[str, str]) -> object:
    """Recursively replace only graph-link node identities."""

    if isinstance(value, dict):
        return {key: _replace_links(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        if (
            len(value) == 2
            and isinstance(value[0], str)
            and value[0] in mapping
            and isinstance(value[1], int)
        ):
            return [mapping[value[0]], value[1]]
        return [_replace_links(item, mapping) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        f"Unsupported Prompt Control expansion value: {type(value).__name__}."
    )
