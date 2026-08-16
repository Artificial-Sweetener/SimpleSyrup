# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Benchmark warmed native and all-one regional SDXL LoRA execution."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.sdxl_attention_couple_parity.cases import load_parity_case
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_regional_lora_performance.runner import run_steady_state_comparison

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\universal-regional-adapter\sdxl-steady-state"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Load external controls and execute the matched steady-state benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--prompt-case", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, default=Path(r"<COMFY_ROOT>"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = IntegrationArtifacts(args.output_root)
    try:
        prompts = load_parity_case(args.prompt_case)
        result = run_steady_state_comparison(
            artifacts,
            inventory=SdxlVisualInventory.load(args.inventory),
            comfy_root=args.comfy_root,
            base_positive_g="1girl, posing, mature female",
            base_positive_l="1girl, posing, mature female",
            negative_g=_solo_negative(prompts.base_negative_g),
            negative_l=_solo_negative(prompts.base_negative_l),
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("SDXL steady-state benchmark failed at %s", artifacts.root)
        return 1
    LOGGER.info("SDXL steady-state benchmark completed: %s", result)
    return 0


def _solo_negative(prompt: str) -> str:
    """Remove only exact two-person-only negative tokens."""

    return ", ".join(
        segment.strip()
        for segment in prompt.split(",")
        if segment.strip().casefold() not in {"1girl", "solo"}
    )


if __name__ == "__main__":
    raise SystemExit(main())
