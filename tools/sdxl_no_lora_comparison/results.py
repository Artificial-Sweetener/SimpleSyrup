# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persist the complete matched SDXL no-LoRA comparison result."""

from __future__ import annotations

import json
from pathlib import Path

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.sampling_controls import (
    SDXL_VISUAL_SAMPLING,
)
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)

from .workflow import SdxlNoLoraComparisonWorkflows


class SdxlNoLoraResultRecorder:
    """Own timing derivation and final comparison serialization."""

    def __init__(self, root: Path) -> None:
        """Retain the existing managed artifact directory."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("No-LoRA result root must already exist.")

    def record(
        self,
        *,
        workflows: SdxlNoLoraComparisonWorkflows,
        prompts: SdxlVisualPromptSet,
        plain: JsonObject,
        regional: JsonObject,
        system_stats: JsonObject,
        port: int,
    ) -> Path:
        """Write one complete result after every managed owner is clean."""

        if not isinstance(workflows, SdxlNoLoraComparisonWorkflows):
            raise TypeError("No-LoRA result requires matched workflows.")
        if not isinstance(prompts, SdxlVisualPromptSet):
            raise TypeError("No-LoRA result requires its prompt fixture.")
        if isinstance(port, bool) or not isinstance(port, int) or port < 1:
            raise ValueError("No-LoRA result requires a valid managed port.")
        plain_model_ms = _metric_float(plain, "model_runtime_ms")
        regional_model_ms = _metric_float(regional, "model_runtime_ms")
        ratio = regional_model_ms / plain_model_ms
        result = self._root / "sdxl-no-lora-comparison.json"
        result.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "sampling": {
                        "seed": SDXL_VISUAL_SAMPLING.seed,
                        "steps": SDXL_VISUAL_SAMPLING.steps,
                        "cfg": SDXL_VISUAL_SAMPLING.cfg,
                        "sampler": SDXL_VISUAL_SAMPLING.sampler,
                        "scheduler": SDXL_VISUAL_SAMPLING.scheduler,
                        "width": 1024,
                        "height": 1024,
                    },
                    "prompts": _prompt_evidence(workflows, prompts),
                    "observations": {"plain": plain, "regional": regional},
                    "regional_to_plain_model_ratio": ratio,
                    "regional_model_overhead_percent": (ratio - 1.0) * 100.0,
                    "system_stats": system_stats,
                    "cleanup": {
                        "server": True,
                        "model_links": True,
                        "masks": True,
                        "port": port,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return result


def _prompt_evidence(
    workflows: SdxlNoLoraComparisonWorkflows,
    prompts: SdxlVisualPromptSet,
) -> JsonObject:
    """Return exact combined and regional prompt declarations."""

    return {
        "plain_positive_g": workflows.plain_positive_g,
        "plain_positive_l": workflows.plain_positive_l,
        "plain_negative_g": workflows.plain_negative_g,
        "plain_negative_l": workflows.plain_negative_l,
        "regional": {
            "base_positive_g": prompts.base_positive_g,
            "base_positive_l": prompts.base_positive_l,
            "base_negative_g": prompts.base_negative_g,
            "base_negative_l": prompts.base_negative_l,
            "left_positive_g": prompts.left_positive_g,
            "left_positive_l": prompts.left_positive_l,
            "left_negative_g": prompts.left_negative_g,
            "left_negative_l": prompts.left_negative_l,
            "right_positive_g": prompts.right_positive_g,
            "right_positive_l": prompts.right_positive_l,
            "right_negative_g": prompts.right_negative_g,
            "right_negative_l": prompts.right_negative_l,
        },
    }


def _metric_float(observation: JsonObject, key: str) -> float:
    """Return one positive recorded timing value."""

    value = observation.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"No-LoRA comparison {key} must be numeric.")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"No-LoRA comparison {key} must be positive.")
    return result
