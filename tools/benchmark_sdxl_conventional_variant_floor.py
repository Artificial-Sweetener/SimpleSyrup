# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Benchmark the warmed conventional two-variant SDXL performance floor."""

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
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_full_strength_lora_fidelity.composition_cases import (
    full_strength_composition_cases,
)
from tools.sdxl_regional_lora_performance.conventional_variant_runner import (
    run_conventional_variant_floor,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_ROOT = default_benchmark_artifact_root(
    "universal-regional-adapter/sdxl-conventional-variant-floor"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Load external fixtures and run the managed floor comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--prompt-case", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, default=default_comfy_root())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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
        case = full_strength_composition_cases(inventory, prompt_set)[2]
        result = run_conventional_variant_floor(
            artifacts,
            inventory=inventory,
            case=case,
            comfy_root=args.comfy_root,
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("Conventional variant floor failed at %s", artifacts.root)
        return 1
    LOGGER.info("Conventional variant floor completed: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
