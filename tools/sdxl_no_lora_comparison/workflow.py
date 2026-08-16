# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build matched plain and regional SDXL no-LoRA visual workflows."""

from __future__ import annotations

from dataclasses import dataclass

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_attention_coupling_integration.visual_workflow import (
    BuiltSdxlVisualWorkflow,
    build_sdxl_visual_workflow,
)
from tools.sdxl_full_strength_lora_fidelity.plain_workflow import (
    BuiltPlainKSamplerWorkflow,
    build_plain_ksampler_workflow,
)


@dataclass(frozen=True, slots=True)
class SdxlNoLoraComparisonWorkflows:
    """Retain matched plain and regional image-producing graphs."""

    plain: BuiltPlainKSamplerWorkflow
    regional: BuiltSdxlVisualWorkflow
    plain_positive_g: str
    plain_positive_l: str
    plain_negative_g: str
    plain_negative_l: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return the union required by both graph families."""

        return self.plain.required_node_ids | self.regional.required_node_ids


def build_no_lora_comparison_workflows(
    *,
    run_id: str,
    checkpoint_name: str,
    mask_names: tuple[str, str],
    prompts: SdxlVisualPromptSet,
) -> SdxlNoLoraComparisonWorkflows:
    """Build same-control vanilla and regional no-LoRA graphs."""

    if not run_id.strip() or not checkpoint_name.strip():
        raise ValueError("No-LoRA comparison identities must be non-empty.")
    positive_g = _combine(
        prompts.base_positive_g,
        prompts.left_positive_g,
        prompts.right_positive_g,
    )
    positive_l = _combine(
        prompts.base_positive_l,
        prompts.left_positive_l,
        prompts.right_positive_l,
    )
    negative_g = _combine(
        prompts.base_negative_g,
        prompts.left_negative_g,
        prompts.right_negative_g,
    )
    negative_l = _combine(
        prompts.base_negative_l,
        prompts.left_negative_l,
        prompts.right_negative_l,
    )
    plain = build_plain_ksampler_workflow(
        run_id=f"{run_id}-plain",
        checkpoint_name=checkpoint_name,
        positive_g=positive_g,
        positive_l=positive_l,
        negative_g=negative_g,
        negative_l=negative_l,
    )
    regional_case = prompts.control_case(
        "regional-no-lora",
        "SimpleSyrup regional — pink/black controls — no LoRA",
        regional_prompt_weight=1.0,
    )
    regional = build_sdxl_visual_workflow(
        run_id=f"{run_id}-regional",
        checkpoint_name=checkpoint_name,
        mask_names=mask_names,
        case=regional_case,
    )
    return SdxlNoLoraComparisonWorkflows(
        plain,
        regional,
        positive_g,
        positive_l,
        negative_g,
        negative_l,
    )


def without_image_outputs(prompt: dict[str, JsonObject]) -> dict[str, JsonObject]:
    """Remove decode/save terminals while retaining synchronized sampling evidence."""

    return {
        node_id: node
        for node_id, node in prompt.items()
        if node["class_type"] not in {"VAEDecode", "SaveImage"}
    }


def _combine(*segments: str) -> str:
    """Join authored prompt sections without separator syntax."""

    if any(not segment.strip() for segment in segments):
        raise ValueError("No-LoRA comparison prompt sections must be non-empty.")
    return ", ".join(segment.strip() for segment in segments)
