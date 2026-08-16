# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Project one exact SDXL visual graph into an image-free diagnostic graph."""

from __future__ import annotations

from dataclasses import dataclass

from tools.comfy_api import JsonObject

from .visual_case_model import SdxlVisualCase, VisualMode
from .visual_workflow import build_sdxl_visual_workflow


@dataclass(frozen=True, slots=True)
class BuiltSdxlVisualDiagnosticWorkflow:
    """Expose the image-free prompt and its exact evidence terminals."""

    prompt: dict[str, JsonObject]
    metrics_node_id: str
    diagnostics_node_id: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every live Comfy node required by the diagnostic graph."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


def build_sdxl_visual_diagnostic_workflow(
    *,
    run_id: str,
    checkpoint_name: str,
    mask_names: tuple[str, str],
    case: SdxlVisualCase,
) -> BuiltSdxlVisualDiagnosticWorkflow:
    """Remove only decode/save work from one exact full-mode visual graph."""

    if not isinstance(case, SdxlVisualCase):
        raise TypeError("SDXL visual diagnostics require a visual case.")
    if case.modes != (VisualMode.FULL,):
        raise ValueError("SDXL visual diagnostics require exactly full mode.")
    built = build_sdxl_visual_workflow(
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        mask_names=mask_names,
        case=case,
    )
    if len(built.outputs) != 1:
        raise ValueError("SDXL visual diagnostics require one full output.")
    terminal = built.outputs[0].outputs
    prompt = {
        node_id: node
        for node_id, node in built.prompt.items()
        if node["class_type"] not in {"VAEDecode", "SaveImage"}
    }
    if (
        terminal.metrics_node_id not in prompt
        or terminal.diagnostics_node_id not in prompt
    ):
        raise ValueError("SDXL visual diagnostic terminals were removed.")
    return BuiltSdxlVisualDiagnosticWorkflow(
        prompt,
        terminal.metrics_node_id,
        terminal.diagnostics_node_id,
    )
