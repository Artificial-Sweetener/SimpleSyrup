# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build one sampler-free regional materialization parity workflow."""

from __future__ import annotations

from dataclasses import dataclass

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.graph import NodeReference
from tools.sdxl_attention_coupling_integration.visual_case_model import SdxlVisualCase
from tools.sdxl_regional_lora_performance.two_adapter_workflow import (
    TwoAdapterPerformanceMode,
    prepare_two_adapter_sampling,
)

_PARITY_NODE = "SimpleSyrupBenchmark.CompareMaterializationParity"


@dataclass(frozen=True, slots=True)
class BuiltMaterializationParityWorkflow:
    """Expose one immutable image-free graph and its terminal identity."""

    prompt: dict[str, JsonObject]
    terminal_node_id: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every Comfy node required by the graph."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


def build_materialization_parity_workflow(
    *,
    run_id: str,
    checkpoint_name: str,
    mask_names: tuple[str, str],
    case: SdxlVisualCase,
) -> BuiltMaterializationParityWorkflow:
    """Build the accepted two-regional-LoRA request without a sampler."""

    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Materialization parity run id must be non-empty.")
    prepared = prepare_two_adapter_sampling(
        checkpoint_name=checkpoint_name,
        mask_names=mask_names,
        case=case,
        mode=TwoAdapterPerformanceMode.REGIONAL,
    )
    region_masks = _region_masks(prepared.sampler_extras)
    terminal = prepared.graph.add(
        _PARITY_NODE,
        model=prepared.model,
        positive=prepared.positive,
        negative=prepared.negative,
        region_masks=region_masks,
        latent_image=prepared.latent,
        region_mask_feather=case.region_mask_feather,
        run_id=run_id,
    )
    return BuiltMaterializationParityWorkflow(prepared.graph.prompt, terminal)


def _region_masks(extras: dict[str, object]) -> NodeReference:
    """Return the exact mask reference retained by regional preparation."""

    value = extras.get("region_masks")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not isinstance(value[0], str)
        or isinstance(value[1], bool)
        or not isinstance(value[1], int)
    ):
        raise ValueError("Materialization parity graph lost its region masks.")
    return value
