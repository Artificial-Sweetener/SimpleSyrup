# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Project fidelity graphs into image-free SDXL performance workflows."""

from __future__ import annotations

from dataclasses import dataclass

from tools.comfy_api import JsonObject
from tools.sdxl_full_strength_lora_fidelity.cases import FullStrengthFidelityCase
from tools.sdxl_full_strength_lora_fidelity.workflow import build_fidelity_workflow


@dataclass(frozen=True, slots=True)
class BuiltSdxlRegionalLoraPerformanceWorkflow:
    """Expose one image-free graph and its exact metric terminal."""

    prompt: dict[str, JsonObject]
    metrics_node_id: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every live Comfy node required by the projected graph."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


def build_performance_workflow(
    *,
    run_id: str,
    checkpoint_name: str,
    mask_name: str,
    base_positive_g: str,
    base_positive_l: str,
    negative_g: str,
    negative_l: str,
    case: FullStrengthFidelityCase,
) -> BuiltSdxlRegionalLoraPerformanceWorkflow:
    """Remove decode/save work from one otherwise identical fidelity graph."""

    built = build_fidelity_workflow(
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        mask_name=mask_name,
        base_positive_g=base_positive_g,
        base_positive_l=base_positive_l,
        negative_g=negative_g,
        negative_l=negative_l,
        case=case,
    )
    prompt = {
        node_id: node
        for node_id, node in built.prompt.items()
        if node["class_type"] not in {"SaveImage", "VAEDecode"}
    }
    return BuiltSdxlRegionalLoraPerformanceWorkflow(prompt, built.metrics_node_id)
