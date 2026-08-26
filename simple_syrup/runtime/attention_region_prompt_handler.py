# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Register pre-validation prompt rewriting for downstream attention nodes."""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from importlib import import_module as _import_module
from typing import Any, cast

from .attention_region_graph import (
    ATTENTION_REGION_PROMPT_PLANNER,
    ATTENTION_REGION_PROMPT_REWRITER,
)
from .attention_region_store import ATTENTION_REGION_CAPTURE_STORE
from .comfy_graph_provenance import NodeRegistry

LOGGER = logging.getLogger(__name__)
_REGISTRATION_MARKER = "_simple_syrup_attention_region_prompt_handler"


def register_attention_region_prompt_handler() -> None:
    """Register exactly one graph-intent handler when PromptServer is available."""

    try:
        server_module = _import_module("server")
    except ImportError:
        return
    prompt_server = getattr(server_module, "PromptServer", None)
    instance = getattr(prompt_server, "instance", None)
    if instance is None or getattr(instance, _REGISTRATION_MARKER, False):
        return
    registrar = getattr(instance, "add_on_prompt_handler", None)
    if not callable(registrar):
        return
    registrar(_rewrite_attention_region_prompt)
    setattr(instance, _REGISTRATION_MARKER, True)


def _rewrite_attention_region_prompt(json_data: object) -> object:
    """Plan and inject attention capture while preserving invalid submissions."""

    if not isinstance(json_data, MutableMapping):
        return json_data
    prompt = json_data.get("prompt")
    if not isinstance(prompt, MutableMapping):
        return json_data
    try:
        ATTENTION_REGION_CAPTURE_STORE.clear()
        plans = ATTENTION_REGION_PROMPT_PLANNER.build(
            cast(dict[str, Any], prompt),
            _node_registry(),
        )
        ATTENTION_REGION_PROMPT_REWRITER.rewrite(
            cast(dict[str, Any], prompt),
            plans,
        )
    except (TypeError, ValueError, RuntimeError):
        LOGGER.exception("Attention-region prompt rewrite failed")
    return json_data


def _node_registry() -> NodeRegistry:
    """Return Comfy's active node registry after extension registration."""

    nodes_module = _import_module("nodes")
    registry = getattr(nodes_module, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(registry, dict):
        raise TypeError("Comfy node registry is unavailable for attention tracing.")
    return cast(NodeRegistry, registry)
