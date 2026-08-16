# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run paired global and all-one regional full-strength SDXL LoRA references."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.sdxl_attention_couple_parity.cases import load_parity_case
from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    LEFT_CHARACTER_SELECTION,
    RIGHT_CHARACTER_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_full_strength_lora_fidelity.cases import (
    CharacterFidelitySpec,
    fidelity_cases,
)
from tools.sdxl_full_strength_lora_fidelity.execution import execute_fidelity_cases

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = Path(r"<COMFY_ROOT>")
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\universal-regional-adapter\full-strength-fidelity"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Load external controls and execute exactly four labeled artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--prompt-case", type=Path, required=True)
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
        solo_negative_g = _without_solo_rejections(prompts.base_negative_g)
        solo_negative_l = _without_solo_rejections(prompts.base_negative_l)
        cases = fidelity_cases(
            (
                CharacterFidelitySpec(
                    "left-character",
                    inventory.left_character.label,
                    LEFT_CHARACTER_SELECTION,
                    inventory.left_character.prompt_g,
                    inventory.left_character.prompt_l,
                ),
                CharacterFidelitySpec(
                    "right-character",
                    inventory.right_character.label,
                    RIGHT_CHARACTER_SELECTION,
                    inventory.right_character.prompt_g,
                    inventory.right_character.prompt_l,
                ),
            )
        )
        result = execute_fidelity_cases(
            artifacts,
            inventory=inventory,
            cases=cases,
            comfy_root=args.comfy_root,
            base_positive_g="1girl, posing, mature female",
            base_positive_l="1girl, posing, mature female",
            negative_g=solo_negative_g,
            negative_l=solo_negative_l,
            readiness_timeout=args.readiness_timeout,
            prompt_timeout=args.prompt_timeout,
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("Full-strength fidelity run failed at %s", artifacts.root)
        return 1
    LOGGER.info("Full-strength fidelity run completed: %s", result)
    return 0


def _without_solo_rejections(prompt: str) -> str:
    """Remove two-person-only rejections from a solo fidelity control."""

    retained = tuple(
        segment.strip()
        for segment in prompt.split(",")
        if segment.strip().casefold() not in {"1girl", "solo"}
    )
    if not retained:
        raise ValueError("Solo fidelity negative prompt cannot be empty.")
    return ", ".join(retained)


if __name__ == "__main__":
    raise SystemExit(main())
