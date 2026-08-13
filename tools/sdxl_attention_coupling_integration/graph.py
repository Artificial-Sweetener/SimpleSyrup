# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own deterministic API graph construction for managed SDXL workflows."""

from __future__ import annotations

from typing import TypeAlias

from tools.comfy_api import JsonObject

NodeReference: TypeAlias = list[str | int]


class SdxlWorkflowGraph:
    """Own deterministic numeric API node identities."""

    def __init__(self) -> None:
        """Initialize one empty prompt graph."""

        self.prompt: dict[str, JsonObject] = {}

    def add(self, class_type: str, **inputs: object) -> str:
        """Append one node and return its stable numeric identity."""

        node_id = str(len(self.prompt) + 1)
        self.prompt[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id
