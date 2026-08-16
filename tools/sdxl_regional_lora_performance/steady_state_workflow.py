# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Represent cache-realistic seed-revisable SDXL timing workflows."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from tools.comfy_api import JsonObject


@dataclass(frozen=True, slots=True)
class BuiltSdxlSteadyStateWorkflow:
    """Expose one stable prepared graph and its synchronized terminal."""

    prompt: dict[str, JsonObject]
    sampler_node_id: str
    completion_node_id: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every live Comfy node required by this graph."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())

    def prompt_for_seed(self, seed: int) -> dict[str, JsonObject]:
        """Copy the stable graph while changing only its sampler seed."""

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(
                "Steady-state sampler seed must be a non-negative integer."
            )
        prompt = copy.deepcopy(self.prompt)
        sampler = prompt.get(self.sampler_node_id)
        if not isinstance(sampler, dict):
            raise ValueError("Steady-state workflow is missing its sampler node.")
        inputs = sampler.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("Steady-state sampler inputs are invalid.")
        inputs["seed"] = seed
        return prompt
