# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Capture one exact image-free SDXL regional-LoRA diagnostic trajectory."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.default_paths import (
    default_benchmark_artifact_root,
    default_comfy_root,
)
from tools.sdxl_attention_couple_parity.cases import load_parity_case
from tools.sdxl_attention_coupling_integration.visual_case_selection import (
    select_visual_cases,
)
from tools.sdxl_attention_coupling_integration.visual_diagnostic_execution import (
    execute_sdxl_visual_diagnostics,
)
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_full_strength_lora_fidelity.composition_cases import (
    full_strength_composition_cases,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = default_comfy_root()
DEFAULT_OUTPUT_ROOT = default_benchmark_artifact_root(
    "universal-regional-adapter/sdxl-visual-diagnostics"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Load external fixtures and capture one selected full-mode trajectory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--prompt-case", type=Path, required=True)
    parser.add_argument("--case-id", default="full-strength-left-only")
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--readiness-timeout", type=float, default=240.0)
    parser.add_argument("--prompt-timeout", type=float, default=1200.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = IntegrationArtifacts(args.output_root)
    try:
        inventory = SdxlVisualInventory.load(args.inventory)
        prompts = load_parity_case(args.prompt_case)
        prompt_set = SdxlVisualPromptSet(
            base_positive_g=prompts.base_positive_g,
            base_positive_l=prompts.base_positive_l,
            base_negative_g=prompts.base_negative_g,
            base_negative_l=prompts.base_negative_l,
            left_positive_g=prompts.left_positive_g,
            left_positive_l=prompts.left_positive_l,
            right_positive_g=prompts.right_positive_g,
            right_positive_l=prompts.right_positive_l,
            left_negative_g=prompts.left_negative_g,
            left_negative_l=prompts.left_negative_l,
            right_negative_g=prompts.right_negative_g,
            right_negative_l=prompts.right_negative_l,
        )
        selected = select_visual_cases(
            full_strength_composition_cases(inventory, prompt_set),
            (args.case_id,),
        )
        result = execute_sdxl_visual_diagnostics(
            artifacts,
            inventory=inventory,
            case=selected[0],
            comfy_root=args.comfy_root,
            readiness_timeout=args.readiness_timeout,
            prompt_timeout=args.prompt_timeout,
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("SDXL visual diagnostics failed at %s", artifacts.root)
        return 1
    LOGGER.info("SDXL visual diagnostics completed: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
