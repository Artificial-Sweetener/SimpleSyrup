# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build a seed- and capture-revisable SDXL cold-attribution workflow."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.visual_case_model import SdxlVisualCase

from .sampling_workflow import add_sdxl_sampler
from .two_adapter_workflow import (
    TwoAdapterPerformanceMode,
    prepare_two_adapter_sampling,
)

_CAPTURE_NODE = "SimpleSyrupBenchmark.CaptureColdPathDiagnostics"
_READ_NODE = "SimpleSyrupBenchmark.ReadColdPathDiagnostics"


@dataclass(frozen=True, slots=True)
class BuiltSdxlColdPathWorkflow:
    """Expose one stable graph with explicit execution identity inputs."""

    prompt: dict[str, JsonObject]
    sampler_node_id: str
    capture_node_id: str
    terminal_node_id: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every Comfy node required by the graph."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())

    def prompt_for_execution(
        self,
        *,
        seed: int,
        run_id: str,
    ) -> dict[str, JsonObject]:
        """Change only sampler seed and the paired capture identity."""

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Cold-path sampler seed must be a non-negative integer.")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Cold-path execution run id must be non-empty.")
        prompt = copy.deepcopy(self.prompt)
        _inputs(prompt, self.sampler_node_id)["seed"] = seed
        _inputs(prompt, self.capture_node_id)["run_id"] = run_id
        _inputs(prompt, self.terminal_node_id)["run_id"] = run_id
        return prompt


def build_sdxl_cold_path_workflow(
    *,
    checkpoint_name: str,
    mask_names: tuple[str, str],
    case: SdxlVisualCase,
) -> BuiltSdxlColdPathWorkflow:
    """Build one image-free two-regional-LoRA attribution graph."""

    prepared = prepare_two_adapter_sampling(
        checkpoint_name=checkpoint_name,
        mask_names=mask_names,
        case=case,
        mode=TwoAdapterPerformanceMode.REGIONAL,
    )
    capture = prepared.graph.add(
        _CAPTURE_NODE,
        model=prepared.model,
        run_id="cold-path-template",
    )
    sampler = add_sdxl_sampler(prepared, model=[capture, 0])
    terminal = prepared.graph.add(
        _READ_NODE,
        latent=[sampler, 0],
        run_id="cold-path-template",
    )
    return BuiltSdxlColdPathWorkflow(
        prompt=prepared.graph.prompt,
        sampler_node_id=sampler,
        capture_node_id=capture,
        terminal_node_id=terminal,
    )


def _inputs(prompt: dict[str, JsonObject], node_id: str) -> dict[str, object]:
    """Return one validated mutable API node input mapping."""

    node = prompt.get(node_id)
    if not isinstance(node, dict):
        raise ValueError(f"Cold-path workflow is missing node {node_id!r}.")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"Cold-path workflow node {node_id!r} has invalid inputs.")
    return inputs
